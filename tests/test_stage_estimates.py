"""Shape-only cost estimators for stages (never read data, never raise)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dfxm.config.models import CostEstimate

H, W = 8, 16
CCMTH_PATH = "/entry/ccmth/Center of mass/Center of mass"
MOSA_PATHS = {
    "chi_com_path": "/entry/chi/Center of mass/Center of mass",
    "chi_fwhm_path": "/entry/chi/FWHM/FWHM",
    "mu_com_path": "/entry/mu/Center of mass/Center of mass",
    "mu_fwhm_path": "/entry/mu/FWHM/FWHM",
}


def _make_layers(tmp_path, n_layers=3, dtype="float32", *, mosa=False):
    """A root with *n_layers* ``layer__N`` folders, each holding a maps.h5."""
    root = tmp_path / "root"
    root.mkdir()
    for i in range(n_layers):
        folder = root / f"layer__{i + 1}"
        folder.mkdir()
        with h5py.File(folder / "maps.h5", "w") as f:
            layer = np.zeros((H, W), dtype=dtype)
            paths = MOSA_PATHS.values() if mosa else (CCMTH_PATH,)
            for path in paths:
                f.create_dataset(path, data=layer)
    return str(root)


def _strain_params(root, **over):
    params = {
        "mode": "batch",
        "root_folder": root,
        "folder_pattern": "layer__*",
        "maps_filename": "maps.h5",
        "ccmth_com_path": CCMTH_PATH,
    }
    params.update(over)
    return params


def test_strain_estimate_reports_shape_and_peak(tmp_path):
    from dfxm.stages.strain import estimate

    root = _make_layers(tmp_path, n_layers=3, dtype="float32")
    est = estimate(_strain_params(root))
    assert isinstance(est, CostEstimate)
    assert est.shape == (3, H, W)
    assert est.input_bytes == 3 * H * W * 4
    # run() holds a float64 map per layer AND the np.stack copy simultaneously
    assert est.peak_bytes == 2 * 3 * H * W * 8


def test_strain_estimate_sizes_one_layer_not_all_of_them(tmp_path, monkeypatch):
    """It must open the first maps.h5 only — this runs on every form change."""
    from dfxm.stages import strain

    root = _make_layers(tmp_path, n_layers=5)
    opened = []
    real_open = h5py.File

    def counting_open(name, *a, **k):
        opened.append(str(name))
        return real_open(name, *a, **k)

    monkeypatch.setattr(strain.h5py, "File", counting_open)
    strain.estimate(_strain_params(root))
    assert len(opened) == 1, f"opened {len(opened)} files: {opened}"


def test_mosaicity_estimate_accounts_for_all_four_datasets(tmp_path):
    """run() holds chi/mu x com/fwhm at once, then np.stack adds one more."""
    from dfxm.stages.mosaicity import estimate

    root = _make_layers(tmp_path, n_layers=3, dtype="float32", mosa=True)
    params = {
        "mode": "batch",
        "root_folder": root,
        "folder_pattern": "layer__*",
        "maps_filename": "maps.h5",
        **MOSA_PATHS,
    }
    est = estimate(params)
    per_volume = 3 * H * W * 4
    assert est.input_bytes == 4 * per_volume
    assert est.peak_bytes == 5 * per_volume  # four collected + one stacked


def test_estimators_never_raise_on_a_missing_root(tmp_path):
    from dfxm.stages.mosaicity import estimate as mosa_estimate
    from dfxm.stages.strain import estimate as strain_estimate

    missing = str(tmp_path / "nope")
    for fn in (strain_estimate, mosa_estimate):
        est = fn({"mode": "batch", "root_folder": missing, "folder_pattern": "*"})
        assert est.peak_bytes == 0
        assert est.shape is None
        assert est.note  # says why it is unknown


def test_estimators_never_read_data(tmp_path, monkeypatch):
    """Guard the cheapness contract: shapes only, so it can run on every keystroke."""
    from dfxm.stages.strain import estimate

    root = _make_layers(tmp_path, n_layers=2)

    def explode(*a, **k):
        raise AssertionError("estimator read dataset contents")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", explode)
    est = estimate(_strain_params(root))
    assert est.shape == (2, H, W)


def test_specs_declare_their_estimators():
    from dfxm.stages import mosaicity, strain

    for module in (strain, mosaicity):
        assert module.STAGE.estimate is not None
        assert callable(module.STAGE.estimator())


@pytest.mark.parametrize("dtype,itemsize", [("float32", 4), ("float64", 8), ("uint16", 2)])
def test_strain_input_bytes_follow_the_source_dtype(tmp_path, dtype, itemsize):
    """input_bytes tracks the file; peak is always float64 because run() converts."""
    from dfxm.stages.strain import estimate

    root = _make_layers(tmp_path, n_layers=2, dtype=dtype)
    est = estimate(_strain_params(root))
    assert est.input_bytes == 2 * H * W * itemsize
    assert est.peak_bytes == 2 * 2 * H * W * 8


ALL_ESTIMATOR_STAGES = (
    "strain",
    "mosaicity",
    "slices",
    "rocking",
    "matched",
    "paraview",
    "visualize",
)


@pytest.mark.parametrize("stage_name", ALL_ESTIMATOR_STAGES)
def test_every_volume_stage_declares_an_estimator(stage_name):
    import importlib

    module = importlib.import_module(f"dfxm.stages.{stage_name}")
    assert module.STAGE.estimate == f"dfxm.stages.{stage_name}:estimate"
    assert callable(module.STAGE.estimator())


@pytest.mark.parametrize("stage_name", ALL_ESTIMATOR_STAGES)
def test_every_estimator_survives_junk_params(stage_name):
    """Called on every form change, including while the user is mid-typing."""
    import importlib

    module = importlib.import_module(f"dfxm.stages.{stage_name}")
    junk = (
        {},
        {"raw_root": "", "mosa_volume_file": ""},
        {"raw_root": "/nonexistent", "mosa_volume_file": "/nonexistent/x.h5"},
    )
    for params in junk:
        est = module.estimate(params)
        assert isinstance(est, CostEstimate)
        assert est.peak_bytes >= 0


def test_sum_dataset_bytes_walks_nested_groups(tmp_path):
    from dfxm.common.h5io import sum_dataset_bytes

    path = tmp_path / "v.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("chi/Center of mass", data=np.zeros((3, 4, 5), dtype="float64"))
        f.create_dataset("mu/FWHM", data=np.zeros((3, 4, 5), dtype="float32"))
    total, largest, itemsize = sum_dataset_bytes(str(path))
    assert total == 3 * 4 * 5 * 8 + 3 * 4 * 5 * 4
    assert largest == (3, 4, 5)
    assert itemsize == 8  # the largest dataset's itemsize


def test_sum_dataset_bytes_on_a_missing_file_is_zero(tmp_path):
    from dfxm.common.h5io import sum_dataset_bytes

    assert sum_dataset_bytes(str(tmp_path / "nope.h5")) == (0, None, 0)


_SLICES_ALL_TOGGLES_OFF = {
    "include_mosa_com_chi": False,
    "include_mosa_fwhm_chi": False,
    "include_mosa_com_mu": False,
    "include_mosa_fwhm_mu": False,
    "include_strain": False,
    "include_raw_sum": False,
    "include_raw_specific": False,
    "include_mosa_sum": False,
    "include_mosa_specific": False,
}


def test_slices_is_chunkable_and_peaks_at_four_arrays_worth(tmp_path):
    """astype(float64) + shifted canvas + interpolated output — four arrays'
    worth at the peak for a stacked-source volume (the read + 3 float64
    copies ``prepare_volume`` holds for ``mosa_volume_file``/``strain_volume_file``).

    ``chunkable`` is **True**: the flag read False on the claim that "alignment
    is a whole-volume operation", which is wrong — the alignment runs along Z, so
    blocking it along Z is what `align_volume_streamed` does, and
    `map_coordinates(order=1)` reads only the two Z layers bracketing each
    sample. The peak figure itself still models the old whole-volume loop.
    """
    from dfxm.stages.slices import estimate

    path = tmp_path / "mosa.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("chi/Center of mass", data=np.zeros((4, 8, 16), dtype="float32"))
    params = {"mosa_volume_file": str(path), **_SLICES_ALL_TOGGLES_OFF}
    params["include_mosa_com_chi"] = True
    est = estimate(params)
    n = 4 * 8 * 16
    assert est.chunkable is True
    assert est.input_bytes == n * 4
    assert est.peak_bytes == n * 4 + 3 * n * 8


def test_slices_estimate_survives_mid_typed_roi_strings(tmp_path):
    """A half-typed ROI ("10,", "abc") with a readable file must not raise —
    the estimator runs on every keystroke, and the ROI plays no part in the
    sizing arithmetic, so the peak must match the no-ROI case exactly.
    """
    from dfxm.stages.slices import estimate

    path = tmp_path / "mosa.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("chi/Center of mass", data=np.zeros((4, 8, 16), dtype="float32"))
    params = {"mosa_volume_file": str(path), **_SLICES_ALL_TOGGLES_OFF}
    params["include_mosa_com_chi"] = True
    baseline = estimate(params)
    for junk in ("10,", "abc", "1,2,3"):
        est = estimate({**params, "align_roi_x": junk})
        assert isinstance(est, CostEstimate)
        assert est.peak_bytes == baseline.peak_bytes


def test_slices_peak_across_two_volumes_is_the_max_pair_not_the_sum(tmp_path):
    """run() holds at most the current + previous volume, never every volume."""
    from dfxm.stages.slices import estimate

    mosa_path = tmp_path / "mosa.h5"
    with h5py.File(mosa_path, "w") as f:
        f.create_dataset("chi/Center of mass", data=np.zeros((4, 8, 16), dtype="float32"))
    rocking_path = tmp_path / "aligned_rocking.h5"
    with h5py.File(rocking_path, "w") as f:
        f.create_dataset("sum_intensity", data=np.zeros((2, 8, 16), dtype="float32"))

    params = {
        "mosa_volume_file": str(mosa_path),
        "aligned_rocking_file": str(rocking_path),
        **_SLICES_ALL_TOGGLES_OFF,
    }
    params["include_mosa_com_chi"] = True
    params["include_raw_sum"] = True
    est = estimate(params)

    n_a, n_b = 4 * 8 * 16, 2 * 8 * 16  # mosa (stacked, 3 copies), rocking (aligned, 1 copy)
    load_peak_a = n_a * 4 + 3 * n_a * 8
    load_peak_b = n_b * 4 + 1 * n_b * 8
    expected = max(load_peak_a + n_b * 8, load_peak_b + n_a * 8)
    assert est.chunkable is True
    assert est.peak_bytes == expected
    assert est.peak_bytes < load_peak_a + load_peak_b, "peak must not be the sum of both volumes"


def test_matched_is_chunkable(tmp_path):
    """The median needs the whole stack along the FRAME axis only, so an in-plane
    block gives the identical answer and the stage chunks itself. Peak is the
    per-scan astype(float64) + nanmedian's internal copy + pooled/frame working
    set, independent of how many scan folders match — now the ceiling reached
    when a scan fits one block, not the figure every run reaches.
    """
    from dfxm.stages.matched import estimate

    root = tmp_path / "raw"
    scan = root / "rock__1"
    scan.mkdir(parents=True)
    with h5py.File(scan / "rock__1.h5", "w") as f:
        f.create_dataset("1.1/measurement/pco_ff", data=np.zeros((6, 8, 16), dtype="uint16"))
    est = estimate({"raw_root": str(root), "rocking_pattern": "rock__*"})
    scan_elems, frame_elems = 6 * 8 * 16, 8 * 16
    assert est.chunkable is True
    assert est.peak_bytes == scan_elems * (2 + 16) + 12 * frame_elems * 8
    assert est.peak_bytes == 26112


def test_matched_reports_chunkable_on_every_early_return(tmp_path):
    """The three unresolved-input returns must agree with the resolved one.

    A stage whose `chunkable` depends on whether the glob happened to match yet
    would flip `advice.plan_run` between "chunked" and "disk-backed" while the
    user is still typing the path.
    """
    from dfxm.stages.matched import estimate

    for params in (
        {"raw_root": "", "rocking_pattern": "rock__*"},
        {"raw_root": str(tmp_path / "nope"), "rocking_pattern": "rock__*"},
    ):
        assert estimate(params).chunkable is True
    empty = tmp_path / "raw" / "rock__1"
    empty.mkdir(parents=True)
    with h5py.File(empty / "rock__1.h5", "w") as f:
        f.create_dataset("something/else", data=np.zeros((2, 2)))
    assert estimate({"raw_root": str(tmp_path / "raw"), "rocking_pattern": "rock__*"}).chunkable


def test_matched_peak_does_not_grow_with_folder_count(tmp_path):
    """Only one scan is ever resident at a time — a second folder must double
    input_bytes but leave peak_bytes exactly where it was.
    """
    from dfxm.stages.matched import estimate

    root = tmp_path / "raw"
    for i in range(2):
        scan = root / f"rock__{i}"
        scan.mkdir(parents=True)
        with h5py.File(scan / f"rock__{i}.h5", "w") as f:
            f.create_dataset("1.1/measurement/pco_ff", data=np.zeros((6, 8, 16), dtype="uint16"))
    est = estimate({"raw_root": str(root), "rocking_pattern": "rock__*"})
    assert est.peak_bytes == 26112
    assert est.input_bytes == 2 * 6 * 8 * 16 * 2  # 2 scans x 6 frames x uint16


def test_rocking_peak_models_streaming_per_scan(tmp_path):
    """run() streams one scan at a time (process_raw_scan: uint16 + float32
    coexist briefly, `del frames` before the next scan) — it does not hold
    every scan's stack at once.
    """
    from dfxm.stages.rocking import estimate

    root = tmp_path / "raw"
    for i in range(2):
        scan = root / f"rock__{i}"
        scan.mkdir(parents=True)
        with h5py.File(scan / f"rock__{i}.h5", "w") as f:
            f.create_dataset("1.1/measurement/pco_ff", data=np.zeros((3, 8, 16), dtype="uint16"))
    est = estimate({"raw_root": str(root), "rocking_pattern": "rock__*"})
    assert est.input_bytes == 2 * 3 * 8 * 16 * 2  # 2 scans x 3 frames x uint16
    assert est.input_bytes == 1536
    assert est.peak_bytes == 5120


def test_paraview_peak_is_the_max_over_files_not_the_sum(tmp_path):
    """_process_mosaicity/_process_strain are separate calls — their locals
    (including the raw datasets dict) die on return, so the two files'
    peaks don't add.
    """
    from dfxm.stages.paraview import estimate

    mosa_path = tmp_path / "stacked_volumes.h5"
    with h5py.File(mosa_path, "w") as f:
        for name in ("chi/Center of mass", "chi/FWHM", "mu/Center of mass", "mu/FWHM"):
            f.create_dataset(name, data=np.zeros((4, 8, 16), dtype="float64"))
    strain_path = tmp_path / "stacked_strain_volumes.h5"
    with h5py.File(strain_path, "w") as f:
        f.create_dataset("strain", data=np.zeros((4, 8, 16), dtype="float64"))

    est = estimate({"mosa_volume_file": str(mosa_path), "strain_volume_file": str(strain_path)})

    field_elems = 4 * 8 * 16
    mosa_total = 4 * field_elems * 8
    mosa_peak = mosa_total + 2 * (4 * field_elems) * 8 + field_elems * 8
    strain_total = field_elems * 8
    strain_peak = strain_total + 2 * field_elems * 8 + field_elems * 8
    assert est.chunkable is True
    assert est.input_bytes == mosa_total + strain_total
    assert est.peak_bytes == max(mosa_peak, strain_peak) == mosa_peak
    assert est.peak_bytes < mosa_peak + strain_peak, "peak must not be the sum of both files"


def test_visualize_peak_sums_inputs_because_datasets_dict_outlives_the_loop(tmp_path):
    """Unlike paraview, mosaicity + strain share run()'s scope — the mosaicity
    `datasets` dict is never freed before the strain section runs, so the two
    files' input bytes DO add (unlike paraview's max-over-files).
    """
    from dfxm.stages.visualize import estimate

    mosa_path = tmp_path / "stacked_volumes.h5"
    with h5py.File(mosa_path, "w") as f:
        for name in ("chi/Center of mass", "chi/FWHM", "mu/Center of mass", "mu/FWHM"):
            f.create_dataset(name, data=np.zeros((4, 8, 16), dtype="float64"))
    strain_path = tmp_path / "stacked_strain_volumes.h5"
    with h5py.File(strain_path, "w") as f:
        f.create_dataset("strain", data=np.zeros((4, 8, 16), dtype="float64"))

    est = estimate({"mosa_volume_file": str(mosa_path), "strain_volume_file": str(strain_path)})

    field_elems = 4 * 8 * 16
    total = 4 * field_elems * 8 + field_elems * 8
    assert est.chunkable is True
    assert est.input_bytes == total
    assert est.peak_bytes == total + 3 * field_elems * 8


# -----------------------------------------------------------------------------
# scratch_bytes — the disk a chunked median run needs
# -----------------------------------------------------------------------------
_SCRATCH_SAMY = np.linspace(0.0, 0.05, 6)
_SCRATCH_SAMZ = np.linspace(0.0, 0.05, 6)
_SCRATCH_SHAPE = (6, 12, 20)
_SCRATCH_SCALE_X = 0.15


def _volume_file(tmp_path, name="stacked_volumes.h5", dataset="strain", shape=_SCRATCH_SHAPE):
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset, data=np.zeros(shape, dtype=np.float64))
    return str(path)


def _patch_motors(monkeypatch):
    """Give every estimator the same known motors, so the expected size is exact."""
    from dfxm.common import alignment as A

    monkeypatch.setattr(
        A.raster,
        "motor_positions_for_estimate",
        lambda *a, **k: (_SCRATCH_SAMY, _SCRATCH_SAMZ),
    )


def _expected_scratch_bytes(shape=_SCRATCH_SHAPE):
    from dfxm.common import alignment as A

    extent = A.aligned_extent(
        shape,
        _SCRATCH_SAMY,
        _SCRATCH_SAMZ,
        scale_x=_SCRATCH_SCALE_X,
        samy_direction=1,
    )
    elems = extent[0] * extent[1] * extent[2]
    return elems * 8, extent


def test_the_scratch_fixture_actually_inflates():
    """Precondition for the tests below.

    If the fixture's motors ever stop padding, `scratch_bytes` would equal the
    unpadded size and every assertion below would pass while proving nothing
    about the aligned extent — the defect shape this project has hit nineteen
    times.
    """
    _expected, extent = _expected_scratch_bytes()
    assert extent[2] > _SCRATCH_SHAPE[2], "fixture no longer pads along X"


# `paraview` is the ONLY stage that passes `scratch_dir=` to
# `align_volume_streamed`, so it is the only one whose `scratch_bytes` may be
# non-zero. Everything from here to the visualize/slices block is about it —
# and until now it had no test at all: mutating its gate to `if False:` left
# the whole suite green.
_PV_STRAIN_SHAPE = (6, 12, 40)  # deliberately bigger than the mosaicity file


def _pv_params(tmp_path, **over):
    """Both volume files present, both exports on, motors patched by the caller."""
    params = {
        "mosa_volume_file": _volume_file(tmp_path, "mosa.h5", "chi/Center of mass"),
        "strain_volume_file": _volume_file(tmp_path, "strain.h5", "strain", _PV_STRAIN_SHAPE),
        "raw_root": str(tmp_path),
        "mosa_pattern": "*",
        "strain_pattern": "*",
        "pixel_size_x_um": _SCRATCH_SCALE_X,
        "samy_direction": 1,
        "center_method": "median",
        "center_mosa_com": True,
        "center_strain": False,
    }
    params.update(over)
    return params


def test_the_paraview_scratch_fixture_distinguishes_the_two_files():
    """Precondition for the per-file tests below.

    They tell "priced the mosaicity file" from "priced the strain file" by the
    SIZE of the figure, so the two must not be the same size — otherwise a gate
    gone back to keying both files off one toggle would still pass.
    """
    mosa, _e = _expected_scratch_bytes()
    strain, _e = _expected_scratch_bytes(_PV_STRAIN_SHAPE)
    assert strain > mosa > 0


@pytest.mark.parametrize("method,spills", [("median", True), ("mean", False)])
def test_paraview_reports_scratch_only_for_the_median_centring(
    tmp_path, monkeypatch, method, spills
):
    """`mean` is a single pass — `_multipass_scratch` returns None and nothing caches."""
    from dfxm.stages.paraview import estimate

    _patch_motors(monkeypatch)
    expected, _extent = _expected_scratch_bytes()
    est = estimate(_pv_params(tmp_path, center_method=method))
    assert est.scratch_bytes == (expected if spills else 0)


@pytest.mark.parametrize(
    "mosa_on,strain_on,which",
    [
        (True, False, "mosa"),
        (False, True, "strain"),
        (True, True, "both"),
        (False, False, "neither"),
    ],
)
def test_paraview_prices_the_spill_of_whichever_file_actually_caches(
    tmp_path, monkeypatch, mosa_on, strain_on, which
):
    """The gate is per file, and the two files' caches do not add.

    `_process_mosaicity` caches when `center_mosa_com` is set, `_process_strain`
    when `center_strain` is — separately. Keying both off `center_mosa_com` (as
    this estimator did) reported **zero** for a `center_mosa_com=False,
    center_strain=True, center_method="median"` run that really does spill: the
    under-report, which is the dangerous direction for a disk check. The two
    helpers run sequentially and each releases its cache before the next
    returns, so what is needed is the larger, never the sum.
    """
    from dfxm.stages.paraview import estimate

    _patch_motors(monkeypatch)
    mosa_bytes, _e = _expected_scratch_bytes()
    strain_bytes, _e = _expected_scratch_bytes(_PV_STRAIN_SHAPE)
    expected = {
        "mosa": mosa_bytes,
        "strain": strain_bytes,
        "both": max(mosa_bytes, strain_bytes),
        "neither": 0,
    }[which]

    est = estimate(_pv_params(tmp_path, center_mosa_com=mosa_on, center_strain=strain_on))
    assert est.scratch_bytes == expected
    if which == "both":
        assert est.scratch_bytes < mosa_bytes + strain_bytes, "the two caches must not add"


@pytest.mark.parametrize(
    "key,drop,shape",
    [
        ("export_mosaicity", "strain_volume_file", _SCRATCH_SHAPE),
        ("export_strain", "mosa_volume_file", _PV_STRAIN_SHAPE),
    ],
)
def test_paraview_prices_no_spill_for_a_file_it_will_not_export(
    tmp_path, monkeypatch, key, drop, shape
):
    """A file that is not exported is never opened, so it cannot cache.

    Pricing it would let `plan_run` block a run for disk the run never touches —
    the same defect shape as `visualize`/`slices` reporting a spill they cannot
    perform. Only the file under test is present, so the figure is that file's
    alone and the export toggle is the only thing that can move it.
    """
    from dfxm.stages.paraview import estimate

    _patch_motors(monkeypatch)
    params = _pv_params(tmp_path, center_mosa_com=True, center_strain=True)
    params[drop] = ""
    on = estimate({**params, key: True})
    off = estimate({**params, key: False})

    # Precondition: with the export on, this single file really is priced —
    # otherwise "0 when off" would prove nothing.
    assert on.scratch_bytes == _expected_scratch_bytes(shape)[0]
    assert off.scratch_bytes == 0


def test_scratch_bytes_is_zero_without_motors(tmp_path):
    """The no-motor path builds no aligned volume, so it caches nothing."""
    from dfxm.stages.paraview import estimate

    est = estimate(_pv_params(tmp_path, raw_root="", mosa_pattern="", strain_pattern=""))
    assert est.scratch_bytes == 0


# -----------------------------------------------------------------------------
# visualize and slices report ZERO scratch — and that is the truth, not a gap
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["median", "mean", "midrange"])
def test_visualize_never_prices_a_spill_it_cannot_perform(tmp_path, monkeypatch, method):
    """`visualize` passes no `scratch_dir`, so a median run re-reads and uses no disk.

    Pinned because the obvious "fix" is to add the figure back: with
    `plan_run` consulting `scratch_bytes`, a chunked median run on a disk-tight
    machine would then come back **blocked** — refusing a run that touches zero
    disk, which is exactly what "slower, never failed" forbids. The run-side
    half of this pair is
    `test_stage_visualize.py::test_visualize_never_hands_the_alignment_a_scratch_dir`.
    """
    from dfxm.stages.visualize import estimate

    _patch_motors(monkeypatch)
    est = estimate(
        {
            "mosa_volume_file": _volume_file(tmp_path),
            "raw_root": str(tmp_path),
            "mosa_pattern": "*",
            "pixel_size_x_um": _SCRATCH_SCALE_X,
            "samy_direction": 1,
            "center_method": method,
        }
    )
    # Precondition: the estimate really priced this file, so "0 scratch" is a
    # statement about the spill and not about an estimator that bailed early.
    assert est.peak_bytes > 0 and est.note is None
    assert est.scratch_bytes == 0


@pytest.mark.parametrize("method", ["median", "mean", "midrange"])
def test_slices_never_prices_a_spill_it_cannot_perform(tmp_path, monkeypatch, method):
    """Same as above for `slices`, **on the main return**.

    `prepare_volume` calls `align_volume_streamed` with `center_method=None`, so
    the alignment never computes a multi-pass statistic and never caches; slices
    centres itself afterwards, in core.

    The precondition matters. The previous version of this test used a fixture
    whose `mosa_volume_file` held only a `strain` dataset, so `_standard_volumes`
    resolved nothing and all three cases landed on the coarse **fallback**
    return that the sibling test below already covers — leaving the main return
    unchecked, which is how mutating it stayed green. Asserting
    `peak_bytes != input_bytes` keeps this test in the region it names.
    """
    from dfxm.stages.slices import estimate

    _patch_motors(monkeypatch)
    params = {
        "mosa_volume_file": _volume_file(tmp_path, "mosa.h5", "chi/Center of mass"),
        "raw_root": str(tmp_path),
        "mosa_pattern": "*",
        "pixel_size_x_um": _SCRATCH_SCALE_X,
        "samy_direction": 1,
        "center_method": method,
        **_SLICES_ALL_TOGGLES_OFF,
    }
    params["include_mosa_com_chi"] = True
    est = estimate(params)

    # Precondition: the MAIN return, where per-dataset sizing resolved. The
    # fallback returns `input_bytes` for the peak; the main one adds the
    # float64 copies, so the two differ.
    assert est.peak_bytes != est.input_bytes, "not on the main return path"
    assert est.scratch_bytes == 0


def test_slices_reports_no_scratch_on_the_coarse_fallback_return_either(tmp_path, monkeypatch):
    """`estimate` has more than one return; zero must hold on all of them."""
    from dfxm.stages import slices as S

    _patch_motors(monkeypatch)
    params = {
        "mosa_volume_file": _volume_file(tmp_path),
        "raw_root": str(tmp_path),
        "mosa_pattern": "*",
        "pixel_size_x_um": _SCRATCH_SCALE_X,
        "samy_direction": 1,
        "center_method": "median",
    }
    # Force the coarse fallback: nothing resolves per-dataset.
    monkeypatch.setattr(S, "_standard_volumes", lambda *a, **k: [])
    est = S.estimate(params)

    # Precondition: we really are on the fallback return, not the main one.
    assert est.peak_bytes == est.input_bytes, "not on the coarse fallback path"
    assert est.scratch_bytes == 0


def test_alignment_estimators_read_motors_but_never_voxels(tmp_path, monkeypatch):
    """The cheapness contract, as widened by the aligned-extent correction.

    Pricing the alignment chain needs the motor VALUES (one scalar per raw scan
    folder), so "shapes only, never data" is no longer literally true. What must
    still hold is the part that makes `estimate` cheap enough to run on every
    keystroke: it never reads a **voxel**. Reading one 3-D dataset would pull
    gigabytes into the GUI process.

    `test_estimators_never_read_data` covers only `strain`, which does not align,
    and would not have caught this widening. `paraview` is the subject because
    it is now the only estimator that reads motors at all — `visualize` and
    `slices` stopped once their (fictitious) spill figures were removed.
    """
    from dfxm.stages.paraview import estimate

    _patch_motors(monkeypatch)
    real_getitem = h5py.Dataset.__getitem__

    def guard(self, key):
        if self.ndim >= 3:
            raise AssertionError(f"estimator read voxels from a {self.ndim}-D dataset")
        return real_getitem(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guard)
    est = estimate(_pv_params(tmp_path))
    # Precondition: the run actually priced an aligned volume, so the guard had
    # something to catch. Without this the test passes on an estimate that never
    # looked at the file at all.
    expected, _extent = _expected_scratch_bytes()
    assert est.scratch_bytes == expected
