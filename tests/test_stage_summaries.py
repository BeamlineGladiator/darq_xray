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
from dfxm.stages.rocking import RockingProducts, RockingResult
from dfxm.stages.slices import SlicesResult
from dfxm.stages.strain import LayerResult, StrainResult
from dfxm.stages.visualize import DatasetProducts, VisualizeResult
from gui.stage_view import _representative_image, _summarize


def _layer(name="lay1", plots=()):
    return LayerResult(
        name=name, shape=(4, 5), vmin=-1e-4, vmax=1e-4, mean=2e-5, std=5e-5, plots=list(plots)
    )


def _strain_result(**kw):
    kw.setdefault("stacked_path", "/proc/stacked_strain_volumes.h5")
    kw.setdefault("volume_shape", (2, 4, 5))
    kw.setdefault("output_dir", "/proc/strain_maps")
    kw.setdefault("layers", [_layer()])
    return StrainResult(**kw)


# -----------------------------------------------------------------------------
# _summarize: strain / mosaicity behaviour
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


def test_summarize_strain_empty_reports_no_layers_without_none():
    result = StrainResult(output_dir="/proc/strain_maps", skipped=["lay1", "lay2: bad maps.h5"])
    text = _summarize("strain", result)
    assert "no strain layers produced" in text
    assert "lay2: bad maps.h5" in text
    assert "None" not in text


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


def test_summarize_mosaicity_empty_reports_no_layers_without_none():
    text = _summarize("mosaicity", MosaicityResult(skipped=["lay1: no datasets"]))
    assert "no mosaicity layers produced" in text
    assert "lay1: no datasets" in text
    assert "None" not in text


# -----------------------------------------------------------------------------
# _summarize: dispatch
# -----------------------------------------------------------------------------
def test_summarize_dispatches_each_stage_to_its_own_format():
    cases = {
        "concat": (
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
        ),
        "strain": (_strain_result(), "stacked: /proc/stacked_strain_volumes.h5"),
        "mosaicity": (
            MosaicityResult(stacked_path="m.h5", datasets={"chi/FWHM": (1, 2, 3)}, layers=["a"]),
            "chi/FWHM: (1, 2, 3)",
        ),
        "rocking": (
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
        ),
        "visualize": (
            VisualizeResult(
                output_dir="o",
                datasets=[DatasetProducts(name="chi", shape=(1, 2, 3), vmin=0.0, vmax=1.0)],
            ),
            "chi: shape=(1, 2, 3)",
        ),
        "paraview": (
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
        ),
        "slices": (
            SlicesResult(
                output_h5="slices.h5", volume_ids=["v"], slice_names=["xy"], n_planes_total=3
            ),
            "planes: 3",
        ),
        "profiles": (
            ProfilesResult(
                output_dir="o",
                mode="parameter",
                jobs=[ProfileJobResult(name="line1", offset_used_um=0.5, figure="f.png")],
            ),
            "mode: parameter",
        ),
        "matched": (
            MatchedResult(n_strain=3, n_matched=2, n_saved=2, vmin=0.0, vmax=1.0),
            "matched 2/3",
        ),
    }
    for stage_name, (result, expected) in cases.items():
        text = _summarize(stage_name, result)
        assert expected in text, f"{stage_name}: {expected!r} not in {text!r}"


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


def test_representative_image_matched_first_png():
    result = MatchedResult(vmin=0.0, vmax=1.0, pngs=["m1.png", "m2.png"])
    assert _representative_image("matched", result) == "m1.png"


def test_representative_image_profiles_uses_job_figure():
    result = ProfilesResult(jobs=[ProfileJobResult(name="j", offset_used_um=0.0, figure="f.png")])
    assert _representative_image("profiles", result) == "f.png"


def test_representative_image_stage_without_images_returns_none():
    assert _representative_image("concat", ConcatResult()) is None
