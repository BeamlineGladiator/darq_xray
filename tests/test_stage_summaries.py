"""Unit tests for the stage-name-keyed Results summaries in gui.stage_view.

``_summarize`` / ``_representative_image`` are pure module-level functions over
the Qt-free result dataclasses, so they are tested headless (no QApplication).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from dfxm.stages.concat import ConcatFileResult, ConcatResult
from dfxm.stages.matched import MatchedResult
from dfxm.stages.mosaicity import MosaicityResult
from dfxm.stages.paraview import ExportInfo, ParaviewResult
from dfxm.stages.profiles import ProfileJobResult, ProfilesResult
from dfxm.stages.registry import STAGE_TARGETS
from dfxm.stages.rocking import RockingProducts, RockingResult
from dfxm.stages.slices import SlicesResult
from dfxm.stages.strain import LayerResult, StrainResult
from dfxm.stages.visualize import DatasetProducts, VisualizeResult
from gui.stage_view import _IMAGE_PICKERS, _SUMMARIZERS, _representative_image, _summarize


def _layer(name="lay1", plots=()):
    return LayerResult(
        name=name, shape=(4, 5), vmin=-1e-4, vmax=1e-4, mean=2e-5, std=5e-5, plots=list(plots)
    )


def _strain_result(**kw):
    kw.setdefault("stacked_path", "/proc/stacked_strain_volumes.h5")
    kw.setdefault("volume_shape", (2, 4, 5))
    kw.setdefault("layers", [_layer()])
    return StrainResult(**kw)


# -----------------------------------------------------------------------------
# Dispatch tables stay in sync with the stage registry
# -----------------------------------------------------------------------------
def test_dispatch_tables_cover_every_registry_stage():
    assert set(_SUMMARIZERS) == set(STAGE_TARGETS)
    assert set(_IMAGE_PICKERS) <= set(_SUMMARIZERS)


# -----------------------------------------------------------------------------
# _summarize: per-stage behaviour
# -----------------------------------------------------------------------------
def test_summarize_strain_lists_layers_and_volume():
    text = _summarize("strain", _strain_result(layers=[_layer("a"), _layer("b")]))
    assert "layers: 2" in text
    assert "volume: (2, 4, 5)" in text
    assert "stacked: /proc/stacked_strain_volumes.h5" in text
    assert "a: shape=(4, 5)" in text
    assert "b: shape=(4, 5)" in text


def test_summarize_strain_shows_skip_reasons():
    text = _summarize("strain", _strain_result(skipped=["lay2: KeyError('chi')"]))
    assert "lay2: KeyError('chi')" in text
    assert "skipped: 1" not in text


def test_summarize_strain_empty_reports_no_layers():
    result = StrainResult(skipped=["lay1: maps.h5 not found", "lay2: bad maps.h5"])
    text = _summarize("strain", result)
    assert "no strain layers produced" in text
    assert "lay2: bad maps.h5" in text
    assert "volume: None" not in text
    assert "stacked: None" not in text


def test_summarize_mosaicity_shows_skip_reasons():
    result = MosaicityResult(
        stacked_path="/proc/stacked_volumes.h5",
        datasets={"chi/Center of mass": (2, 4, 5)},
        layers=["lay1", "lay2"],
        skipped=["lay3: no datasets"],
    )
    text = _summarize("mosaicity", result)
    assert "layers: 2" in text
    assert "chi/Center of mass: (2, 4, 5)" in text
    assert "lay3: no datasets" in text
    assert "skipped: 1" not in text


def test_summarize_mosaicity_empty_reports_no_layers():
    text = _summarize("mosaicity", MosaicityResult(skipped=["lay1: no datasets"]))
    assert "no mosaicity layers produced" in text
    assert "lay1: no datasets" in text
    assert "stacked: None" not in text


def test_summarize_rocking_empty_reports_no_volumes():
    result = RockingResult(output_dir="/out", skipped=["no rocking scans processed successfully"])
    text = _summarize("rocking", result)
    assert "no rocking volumes produced" in text
    assert "no rocking scans processed successfully" in text
    assert "volume: None" not in text
    assert "aligned: None" not in text
    assert "specific frame: None" not in text


def test_summarize_rocking_omits_unsaved_aligned_path():
    result = RockingResult(
        output_dir="/out",
        aligned_path=None,  # save_aligned_h5 unchecked on a successful run
        volume_shape=(1, 2, 3),
        n_layers_used=1,
        specific_frame_idx=5,
        z_span_um=1.5,
        datasets=[RockingProducts(name="pco_ff", vmin=0.0, vmax=1.0)],
    )
    text = _summarize("rocking", result)
    assert "volume: (1, 2, 3)" in text
    assert "aligned: None" not in text


def test_summarize_matched_empty_reports_no_layers():
    result = MatchedResult(
        output_dir="/out",
        n_strain=12,
        n_matched=5,
        vmin=0.0,
        vmax=1.0,
        skipped=["no rocking image could be loaded"],
    )
    text = _summarize("matched", result)
    assert "no matched layers saved" in text
    assert "matched 5/12" in text
    assert "no rocking image could be loaded" in text
    assert "output: None" not in text


def test_summarize_matched_lists_all_skipped_entries():
    reasons = [f"layer {i}: image load failed" for i in range(7)]
    result = MatchedResult(
        layers_dir="/out/rocking_layers",
        n_strain=9,
        n_matched=2,
        n_saved=2,
        vmin=0.0,
        vmax=1.0,
        skipped=reasons,
    )
    text = _summarize("matched", result)
    for reason in reasons:
        assert reason in text


def test_summarize_slices_empty_reports_no_volumes():
    result = SlicesResult(output_dir="/out", skipped=["no input volumes found / selected"])
    text = _summarize("slices", result)
    assert "no volumes sliced" in text
    assert "no input volumes found / selected" in text
    assert "output: None" not in text


# -----------------------------------------------------------------------------
# _summarize: dispatch
# -----------------------------------------------------------------------------
DISPATCH_CASES = [
    pytest.param(
        "concat",
        ConcatResult(
            files=[
                ConcatFileResult(
                    input_path="in.h5",
                    output_path="out.h5",
                    ok=True,
                    n_entries=2,
                    total_frames=10,
                    n_motors=3,
                    n_varying=1,
                )
            ]
        ),
        "[OK] out.h5",
        id="concat",
    ),
    pytest.param(
        "strain", _strain_result(), "stacked: /proc/stacked_strain_volumes.h5", id="strain"
    ),
    pytest.param(
        "mosaicity",
        MosaicityResult(stacked_path="m.h5", datasets={"chi/FWHM": (1, 2, 3)}, layers=["a"]),
        "chi/FWHM: (1, 2, 3)",
        id="mosaicity",
    ),
    pytest.param(
        "rocking",
        RockingResult(
            output_dir="o",
            aligned_path="aligned.h5",
            volume_shape=(1, 2, 3),
            n_layers_used=1,
            specific_frame_idx=5,
            z_span_um=1.5,
            datasets=[RockingProducts(name="pco_ff", vmin=0.0, vmax=1.0)],
        ),
        "aligned: aligned.h5",
        id="rocking",
    ),
    pytest.param(
        "visualize",
        VisualizeResult(
            output_dir="o",
            datasets=[DatasetProducts(name="chi", shape=(1, 2, 3), vmin=0.0, vmax=1.0)],
        ),
        "chi: shape=(1, 2, 3)",
        id="visualize",
    ),
    pytest.param(
        "paraview",
        ParaviewResult(
            output_dir="o",
            exports=[
                ExportInfo(
                    name="strain",
                    pvti_path="strain.pvti",
                    dimensions_xyz=(1, 2, 3),
                    spacing_um_xyz=(1.0, 1.0, 1.0),
                    origin_um_xyz=(0.0, 0.0, 0.0),
                    n_pieces=1,
                    fields=["strain"],
                )
            ],
        ),
        "strain: strain.pvti",
        id="paraview",
    ),
    pytest.param(
        "slices",
        SlicesResult(output_h5="slices.h5", volume_ids=["v"], slice_names=["xy"], n_planes_total=3),
        "planes: 3",
        id="slices",
    ),
    pytest.param(
        "profiles",
        ProfilesResult(
            output_dir="o",
            mode="parameter",
            jobs=[ProfileJobResult(name="line1", offset_used_um=0.5, figure="f.png")],
        ),
        "mode: parameter",
        id="profiles",
    ),
    pytest.param(
        "matched",
        MatchedResult(layers_dir="ld", n_strain=3, n_matched=2, n_saved=2, vmin=0.0, vmax=1.0),
        "matched 2/3",
        id="matched",
    ),
]


@pytest.mark.parametrize(("stage_name", "result", "expected"), DISPATCH_CASES)
def test_summarize_dispatches_each_stage_to_its_own_format(stage_name, result, expected):
    assert expected in _summarize(stage_name, result)


def test_summarize_unknown_stage_falls_back_to_repr():
    result = _strain_result()
    assert _summarize("nonexistent", result) == repr(result)


# -----------------------------------------------------------------------------
# _representative_image
# -----------------------------------------------------------------------------
def test_representative_image_strain_prefers_strain_png():
    result = _strain_result(layers=[_layer(plots=["a_hist.png", "a_strain.png"])])
    assert _representative_image("strain", result) == "a_strain.png"


def test_representative_image_strain_falls_back_to_first_plot():
    result = _strain_result(layers=[_layer(plots=["a_hist.png", "a_detrend.png"])])
    assert _representative_image("strain", result) == "a_hist.png"


def test_representative_image_strain_empty_returns_none():
    assert _representative_image("strain", StrainResult()) is None


def test_representative_image_rocking_uses_top_view():
    result = RockingResult(
        datasets=[RockingProducts(name="pco_ff", vmin=0.0, vmax=1.0, top_view="top.png")]
    )
    assert _representative_image("rocking", result) == "top.png"


def test_representative_image_visualize_uses_top_view():
    result = VisualizeResult(
        datasets=[
            DatasetProducts(name="chi", shape=(1, 2, 3), vmin=0.0, vmax=1.0, top_view="top.png")
        ]
    )
    assert _representative_image("visualize", result) == "top.png"


def test_representative_image_slices_first_png():
    result = SlicesResult(pngs=["s1.png", "s2.png"])
    assert _representative_image("slices", result) == "s1.png"


def test_representative_image_matched_first_png():
    result = MatchedResult(vmin=0.0, vmax=1.0, pngs=["m1.png", "m2.png"])
    assert _representative_image("matched", result) == "m1.png"


def test_representative_image_profiles_uses_job_figure():
    result = ProfilesResult(jobs=[ProfileJobResult(name="j", offset_used_um=0.0, figure="f.png")])
    assert _representative_image("profiles", result) == "f.png"


def test_representative_image_stage_without_images_returns_none():
    assert _representative_image("concat", ConcatResult()) is None
