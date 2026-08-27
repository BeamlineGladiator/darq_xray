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
    root.mkdir(parents=True)
    for i in range(n_layers):
        folder = root / f"layer__{i + 1}"
        folder.mkdir()
        with h5py.File(folder / "maps.h5", "w") as f:
            layer = np.zeros((H, W), dtype=dtype)
            paths = MOSA_PATHS.values() if mosa else (CCMTH_PATH,)
            for path in paths:
                f.create_dataset(path, data=layer)
    return str(root)


def _make_big_layer(tmp_path, *, ny, nx):
    """One layer large enough for the fixed-scale canvas to be the bigger one."""
    root = tmp_path / "big_root"
    folder = root / "big__1"
    folder.mkdir(parents=True)
    with h5py.File(folder / "maps.h5", "w") as f:
        f.create_dataset(CCMTH_PATH, shape=(ny, nx), dtype="float32")
    return str(root)


@pytest.fixture(scope="module")
def _strain_defaults():
    from dfxm.stages import strain

    return strain.STAGE.defaults()


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


def _strain_peak(layer_elems, *, plots=True, canvas_px=0):
    """The recalibrated strain model, spelled out independently of the stage."""
    from dfxm.common.plotting import AGG_RENDER_BYTES_PER_PIXEL
    from dfxm.stages import strain

    peak = strain.STRAIN_PROCESS_FLOOR_BYTES + strain.STRAIN_ARRAY_BYTES_PER_ELEM * layer_elems
    if plots:
        peak += strain.STRAIN_PLOT_PROCESS_BYTES + strain.STRAIN_PLOT_BYTES_PER_ELEM * layer_elems
        peak += AGG_RENDER_BYTES_PER_PIXEL * canvas_px
    return peak


def test_strain_estimate_reports_shape_and_peak(tmp_path, _strain_defaults):
    """One layer's working set, not the volume: `run()` drops each layer as it goes."""
    from dfxm.stages.strain import _plot_canvas_pixels, estimate

    root = _make_layers(tmp_path, n_layers=3, dtype="float32")
    params = _strain_params(root)
    est = estimate(params)
    assert isinstance(est, CostEstimate)
    assert est.shape == (3, H, W)
    assert est.input_bytes == 3 * H * W * 4
    assert est.peak_bytes == _strain_peak(
        H * W, canvas_px=_plot_canvas_pixels({**_strain_defaults, **params}, (H, W))
    )


def test_strain_peak_does_not_grow_with_the_layer_count(tmp_path):
    """The claim the old model got wrong, and the reason it over-predicted 5.2x.

    `run()` appends each layer to the StackedVolumeFile and `del`s it, so a
    hundred layers cost what three do. input_bytes must still scale, or this
    would pass on an estimator that had stopped counting the input at all.
    """
    from dfxm.stages.strain import estimate

    peaks = set()
    inputs = []
    for n in (1, 3, 32):
        root = _make_layers(tmp_path / f"n{n}", n_layers=n, dtype="float32")
        est = estimate(_strain_params(root))
        peaks.add(est.peak_bytes)
        inputs.append(est.input_bytes)
    assert len(peaks) == 1, f"peak moved with the layer count: {sorted(peaks)}"
    assert inputs == [H * W * 4, 3 * H * W * 4, 32 * H * W * 4]


def test_strain_save_plots_off_drops_the_whole_plotting_term(tmp_path):
    """Turning the product off must move the estimate, or the toggle is cosmetic."""
    from dfxm.stages.strain import estimate

    root = _make_layers(tmp_path, n_layers=2, dtype="float32")
    on = estimate(_strain_params(root, save_plots=True)).peak_bytes
    off = estimate(_strain_params(root, save_plots=False)).peak_bytes
    assert off == _strain_peak(H * W, plots=False)
    assert on > off


def test_strain_prices_the_canvas_a_fixed_scale_style_asks_for(tmp_path):
    """A `scale_um_per_cm` style grew the measured peak from 465 to 1682 MiB.

    The estimator reads the style and sizes the same canvas the builder will,
    through the shared `strain_map_geometry`. Without this term the model
    UNDER-predicts a publication-scale run by 3.6x.
    """
    from dfxm.common.plotting import canvas_pixels, style_from_params
    from dfxm.stages.strain import estimate, strain_map_geometry

    # A layer big enough that the fixed-scale box is the larger canvas: at 8x16
    # px it would not be, and this test would pass on an estimator that ignored
    # the style entirely.
    root = _make_big_layer(tmp_path, ny=600, nx=800)
    plain = _strain_params(root, folder_pattern="big__*")
    styled = _strain_params(root, folder_pattern="big__*", plot_style={"scale_um_per_cm": 4.0})

    p_geom, _ = strain_map_geometry(None, 800 * 0.152, 600 * 0.385)
    s_geom, box = strain_map_geometry(style_from_params(styled), 800 * 0.152, 600 * 0.385)
    assert box is not None, "fixture must reach the fixed-scale path"
    assert canvas_pixels(s_geom, 200) > canvas_pixels(p_geom, 200), "fixture must grow the canvas"

    assert estimate(styled).peak_bytes > estimate(plain).peak_bytes


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


def _mosa_params(root, **over):
    params = {
        "mode": "batch",
        "root_folder": root,
        "folder_pattern": "layer__*",
        "maps_filename": "maps.h5",
        **MOSA_PATHS,
    }
    params.update(over)
    return params


def test_mosaicity_estimate_accounts_for_all_four_datasets(tmp_path):
    """One LAYER of each present dataset, plus the writer's buffers — not a volume."""
    from dfxm.stages import mosaicity
    from dfxm.stages.mosaicity import estimate

    root = _make_layers(tmp_path, n_layers=3, dtype="float32", mosa=True)
    est = estimate(_mosa_params(root))
    assert est.input_bytes == 4 * 3 * H * W * 4
    assert est.peak_bytes == mosaicity.MOSAICITY_PROCESS_FLOOR_BYTES + (
        4 + mosaicity.MOSAICITY_WRITER_LAYERS
    ) * (H * W * 4)


def test_mosaicity_each_present_dataset_costs_exactly_one_layer(tmp_path):
    """Measured at 1.00 / 1.01 / 1.00 layers per dataset — the sharpest number
    in the campaign, so it is pinned as an exact step rather than a bound.
    """
    from dfxm.stages.mosaicity import estimate

    keys = list(MOSA_PATHS)
    peaks = []
    for n_present in (1, 2, 3, 4):
        root = tmp_path / f"ds{n_present}"
        folder = root / "layer__1"
        folder.mkdir(parents=True)
        with h5py.File(folder / "maps.h5", "w") as f:
            for key in keys[:n_present]:
                f.create_dataset(MOSA_PATHS[key], shape=(H, W), dtype="float32")
        peaks.append(estimate(_mosa_params(str(root))).peak_bytes)
    steps = [b - a for a, b in zip(peaks, peaks[1:])]
    assert steps == [H * W * 4] * 3, f"per-dataset step is not one layer: {steps}"


def test_mosaicity_peak_does_not_grow_with_the_layer_count(tmp_path):
    """Measured flat: 176.1 MiB at one layer, 194.1 at four, eight and sixteen.

    The old model multiplied by n_layers and over-predicted the real STO2 run
    by 36x. input_bytes must still scale, or an estimator that had stopped
    counting the input would pass this.
    """
    from dfxm.stages.mosaicity import estimate

    peaks, inputs = set(), []
    for n in (1, 3, 32):
        root = _make_layers(tmp_path / f"n{n}", n_layers=n, dtype="float32", mosa=True)
        est = estimate(_mosa_params(root))
        peaks.add(est.peak_bytes)
        inputs.append(est.input_bytes)
    assert len(peaks) == 1, f"peak moved with the layer count: {sorted(peaks)}"
    assert inputs == [4 * n * H * W * 4 for n in (1, 3, 32)]


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
    est = estimate(_strain_params(root, save_plots=False))
    assert est.input_bytes == 2 * H * W * itemsize
    # peak is dtype-independent: the detrend chain is float64 whatever it reads
    assert est.peak_bytes == _strain_peak(H * W, plots=False)


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


def _raw_scans(root, *, n_scans=1, n_frames=6, ny=8, nx=16, dtype="uint16", samy_step_mm=0.0):
    """*n_scans* pco_ff scan folders under *root*, shapes only (no data written).

    *samy_step_mm* spreads the samy positions so the estimators that price the
    samy X-padding have something to price; 0 (the default) means no padding.
    """
    for i in range(n_scans):
        scan = root / f"rock__{i}"
        scan.mkdir(parents=True)
        with h5py.File(scan / f"rock__{i}.h5", "w") as f:
            f.create_dataset("1.1/measurement/pco_ff", shape=(n_frames, ny, nx), dtype=dtype)
            # Scalar datasets (shape ()), as the real ESRF positioners are — a
            # 1-element array here reads back through a deprecated numpy path.
            f.create_dataset("1.1/instrument/positioners/samy", data=samy_step_mm * i)
            f.create_dataset("1.1/instrument/positioners/samz", data=0.002 * i)
    return str(root)


def _matched_peak(*, n_frames, ny, nx, pad=0, itemsize=2, autoclim=True):
    """The recalibrated matched model, spelled out independently of the stage."""
    from dfxm.stages import matched

    peak = matched.MATCHED_PROCESS_FLOOR_BYTES + min(
        n_frames * ny * nx * (itemsize + matched.MEDIAN_WORKING_SET_PER_ELEMENT),
        matched.MEDIAN_BLOCK_WORKING_SET_BYTES,
    )
    peak += matched.MATCHED_PADDED_BYTES_PER_ELEM * ny * (nx + pad)
    if autoclim:
        peak += matched.MATCHED_CLIM_POOL_BYTES_PER_ELEM * ny * nx
    return peak


def test_matched_is_chunkable(tmp_path):
    """The median needs the whole stack along the FRAME axis only, so an in-plane
    block gives the identical answer and the stage chunks itself.
    """
    from dfxm.stages import matched
    from dfxm.stages.matched import estimate

    root = _raw_scans(tmp_path / "raw")
    est = estimate({"raw_root": root, "rocking_pattern": "rock__*"})
    scan_elems = 6 * 8 * 16
    # This fixture is small enough that the median term is the whole stack, not
    # the block cap — asserted, because on the capped side the (itemsize + per
    # element) factor below would not be exercised at all.
    assert scan_elems * (2 + matched.MEDIAN_WORKING_SET_PER_ELEMENT) < (
        matched.MEDIAN_BLOCK_WORKING_SET_BYTES
    )
    assert est.chunkable is True
    assert est.peak_bytes == _matched_peak(n_frames=6, ny=8, nx=16)


def test_matched_median_term_stops_at_the_block_working_set(tmp_path):
    """`load_pco_ff_frame` bounds one block's working set, so the frame count
    cannot grow the peak past that cap.

    Measured at 512x512: 217.6 / 213.5 / 217.8 / 215.8 MiB for 21 / 51 / 101 /
    201 frames — flat. The old model climbed to 928 MiB over the same span.
    """
    from dfxm.stages import matched
    from dfxm.stages.matched import estimate

    ny = nx = 256
    peaks = []
    for n_frames in (64, 256, 1024):
        root = _raw_scans(tmp_path / f"f{n_frames}", n_frames=n_frames, ny=ny, nx=nx)
        peaks.append(estimate({"raw_root": root, "rocking_pattern": "rock__*"}).peak_bytes)
    # Precondition: the smallest fixture must already exceed the cap, or this
    # would be testing three uncapped points that happen to differ.
    assert 64 * ny * nx * (2 + matched.MEDIAN_WORKING_SET_PER_ELEMENT) > (
        matched.MEDIAN_BLOCK_WORKING_SET_BYTES
    )
    assert len(set(peaks)) == 1, f"peak moved with the frame count: {peaks}"
    assert peaks[0] == _matched_peak(n_frames=64, ny=ny, nx=nx)


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

    one = estimate({"raw_root": _raw_scans(tmp_path / "a"), "rocking_pattern": "rock__*"})
    two = estimate(
        {"raw_root": _raw_scans(tmp_path / "b", n_scans=2), "rocking_pattern": "rock__*"}
    )
    assert two.peak_bytes == one.peak_bytes
    assert two.input_bytes == 2 * one.input_bytes
    assert two.input_bytes == 2 * 6 * 8 * 16 * 2  # 2 scans x 6 frames x uint16


def _rocking_peak(*, n_scans, n_frames, ny, nx, itemsize=2, topview=False, volume_elems=None):
    """The recalibrated rocking model, spelled out independently of the stage.

    *volume_elems* is the ALIGNED volume's element count; ``None`` means the
    no-motor path, where the read shape is the volume shape.
    """
    from dfxm.stages import rocking

    if volume_elems is None:
        volume_elems = n_scans * ny * nx
    peak = (
        rocking.ROCKING_PROCESS_FLOOR_BYTES
        + n_frames * ny * nx * (itemsize + 4)
        + rocking.ROCKING_VOLUME_BYTES_PER_ELEM * volume_elems
    )
    if topview:
        peak += rocking.ROCKING_TOPVIEW_BYTES
    return peak


def test_rocking_peak_models_streaming_per_scan(tmp_path):
    """run() streams one scan at a time (process_raw_scan: source + float32 copy
    coexist, `del frames` before the next scan) — but the accumulated volumes do
    scale with the scan count, which is the term the old model under-charged.
    """
    from dfxm.stages.rocking import estimate

    root = _raw_scans(tmp_path / "raw", n_scans=2, n_frames=3)
    est = estimate({"raw_root": root, "rocking_pattern": "rock__*", "save_topview": False})
    assert est.input_bytes == 2 * 3 * 8 * 16 * 2  # 2 scans x 3 frames x uint16
    assert est.input_bytes == 1536
    assert est.peak_bytes == _rocking_peak(n_scans=2, n_frames=3, ny=8, nx=16)


def test_rocking_volume_term_grows_with_the_scan_count(tmp_path):
    """Each extra scan adds a layer to both accumulated volumes and everything
    the alignment then does to them — measured at 48.5 B per element per scan.
    """
    from dfxm.stages import rocking
    from dfxm.stages.rocking import estimate

    peaks = []
    for n in (2, 4, 8):
        root = _raw_scans(tmp_path / f"s{n}", n_scans=n, n_frames=3)
        peaks.append(
            estimate(
                {"raw_root": root, "rocking_pattern": "rock__*", "save_topview": False}
            ).peak_bytes
        )
    steps = [b - a for a, b in zip(peaks, peaks[1:])]
    per_scan = rocking.ROCKING_VOLUME_BYTES_PER_ELEM * 8 * 16
    assert steps == [2 * per_scan, 4 * per_scan]


def test_rocking_prices_the_roi_it_will_actually_read(tmp_path):
    """`process_raw_scan` reads `det[:, ys:ye, xs:xe]`, so everything downstream
    is ROI-sized. On the real STO2 form (105,1937 / 630,1330 of a 2048x2048
    detector) ignoring the ROI over-stated the priced work by 3.3x.

    `input_bytes` must NOT follow the ROI: it is the data on disk, which a crop
    does not change.
    """
    from dfxm.stages.rocking import estimate

    root = _raw_scans(tmp_path / "raw", n_scans=2, n_frames=3, ny=64, nx=64)
    base = {"raw_root": root, "rocking_pattern": "rock__*", "save_topview": False}
    full = estimate(base)
    cropped = estimate({**base, "roi_x": "16,48", "roi_y": "8,40"})
    assert full.peak_bytes == _rocking_peak(n_scans=2, n_frames=3, ny=64, nx=64)
    assert cropped.peak_bytes == _rocking_peak(n_scans=2, n_frames=3, ny=32, nx=32)
    assert cropped.input_bytes == full.input_bytes


@pytest.mark.parametrize(
    "roi_x,roi_y",
    [("", ""), ("junk", "8,40"), ("16,", "8,40"), ("48,16", "8,40"), ("-5,9999", "8,40")],
)
def test_rocking_roi_never_shrinks_the_estimate_on_a_half_typed_form(tmp_path, roi_x, roi_y):
    """A blank, malformed, inverted or out-of-range ROI falls back to the whole
    axis — the direction run() would actually read — and never raises.
    """
    from dfxm.stages.rocking import estimate

    root = _raw_scans(tmp_path / "raw", n_scans=2, n_frames=3, ny=64, nx=64)
    base = {"raw_root": root, "rocking_pattern": "rock__*", "save_topview": False}
    est = estimate({**base, "roi_x": roi_x, "roi_y": roi_y})
    rows = 32 if roi_y == "8,40" and roi_x not in ("junk", "16,") else 64
    assert est.peak_bytes == _rocking_peak(n_scans=2, n_frames=3, ny=rows, nx=64)


def test_rocking_globs_the_pattern_its_source_scan_actually_processes(tmp_path):
    """`source_scan="mosaicity"` processes the MOSA folders, so the estimate must
    count those. On the real STO2 form the rocking glob matches 709 folders for a
    run that processes 76 — a 9x over-statement of the volume term.
    """
    from dfxm.stages.rocking import estimate

    root = tmp_path / "raw"
    _raw_scans(root, n_scans=6, n_frames=3)
    for i in range(2):
        scan = root / f"mosa__{i}"
        scan.mkdir(parents=True)
        with h5py.File(scan / f"mosa__{i}.h5", "w") as f:
            f.create_dataset("1.1/measurement/pco_ff", shape=(3, 8, 16), dtype="uint16")
    base = {
        "raw_root": str(root),
        "rocking_pattern": "rock__*",
        "mosa_pattern": "mosa__*",
        "save_topview": False,
    }
    # Precondition: the two globs must disagree, or this asserts nothing.
    assert len(list(root.glob("rock__*"))) != len(list(root.glob("mosa__*")))
    from_mosa = estimate({**base, "source_scan": "mosaicity"})
    from_rocking = estimate({**base, "source_scan": "rocking"})
    assert from_mosa.shape[0] == 2
    assert from_rocking.shape[0] == 6
    assert from_mosa.peak_bytes == _rocking_peak(n_scans=2, n_frames=3, ny=8, nx=16)
    assert from_mosa.peak_bytes < from_rocking.peak_bytes


def test_rocking_prices_the_topview_render_which_is_on_by_default(tmp_path):
    """`save_topview` costs a data-independent ~365 MiB (the pyvista/VTK import
    and its render context) and defaults to ON. The old model charged nothing
    for it and under-predicted a default run by 14x.
    """
    from dfxm.stages import rocking
    from dfxm.stages.rocking import estimate

    root = _raw_scans(tmp_path / "raw", n_scans=2, n_frames=3)
    base = {"raw_root": root, "rocking_pattern": "rock__*"}
    off = estimate({**base, "save_topview": False}).peak_bytes
    on = estimate({**base, "save_topview": True}).peak_bytes
    assert on - off == rocking.ROCKING_TOPVIEW_BYTES
    # The default must be the expensive one, or the toggle is priced for a
    # configuration nobody runs.
    assert rocking.STAGE.defaults()["save_topview"] is True
    assert estimate(base).peak_bytes == on


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


# -----------------------------------------------------------------------------
# confidence marking
# -----------------------------------------------------------------------------
# `confidence="conservative"` means "not recalibrated since the streaming
# rewrite" — the GUI renders it as "at most ~N (conservative estimate)". Until
# 2026-08-26 four stages carried it: strain, mosaicity, rocking and matched,
# over-predicting real STO2 by 5.2x and 36x (the two that were measured) and,
# in rocking's case, UNDER-predicting a default form by 14x. The recalibration
# campaign measured all four, so none carries the flag any more and every
# estimator in the pipeline now reports "measured".


def test_cost_estimate_confidence_defaults_to_measured():
    est = CostEstimate(1, 1, None, True)
    assert est.confidence == "measured"


def test_no_estimator_still_marks_itself_conservative(tmp_path):
    """The flag must be re-set deliberately, not left behind by a stale label.

    Each fixture is asserted to reach its stage's *priced* return: the
    unresolved-input early returns never touch the marked `return` statement,
    so a test that landed on one would pass on any value of the flag.
    """
    from dfxm.stages import matched, mosaicity, rocking, strain

    root = _make_layers(tmp_path / "s", n_layers=3, dtype="float32")
    mosa_root = _make_layers(tmp_path / "m", n_layers=3, dtype="float32", mosa=True)
    raw_root = _raw_scans(tmp_path / "raw", n_scans=2, n_frames=3)
    cases = [
        ("strain", strain.estimate(_strain_params(root))),
        ("mosaicity", mosaicity.estimate(_mosa_params(mosa_root))),
        ("rocking", rocking.estimate({"raw_root": raw_root, "rocking_pattern": "rock__*"})),
        ("matched", matched.estimate({"raw_root": raw_root, "rocking_pattern": "rock__*"})),
    ]
    for name, est in cases:
        assert est.peak_bytes > 0, f"{name} fixture did not reach the priced return"
        assert est.confidence == "measured", f"{name} still marks itself conservative"


def test_visualize_estimator_is_not_marked(tmp_path):
    from dfxm.stages.visualize import estimate

    mosa_path = tmp_path / "stacked_volumes.h5"
    with h5py.File(mosa_path, "w") as f:
        for name in ("chi/Center of mass", "chi/FWHM", "mu/Center of mass", "mu/FWHM"):
            f.create_dataset(name, data=np.zeros((4, 8, 16), dtype="float64"))
    strain_path = tmp_path / "stacked_strain_volumes.h5"
    with h5py.File(strain_path, "w") as f:
        f.create_dataset("strain", data=np.zeros((4, 8, 16), dtype="float64"))

    est = estimate({"mosa_volume_file": str(mosa_path), "strain_volume_file": str(strain_path)})
    assert est.peak_bytes > 0, "fixture did not reach the priced return"
    assert est.confidence == "measured"


def test_paraview_estimator_is_not_marked(tmp_path):
    from dfxm.stages.paraview import estimate

    mosa_path = tmp_path / "stacked_volumes.h5"
    with h5py.File(mosa_path, "w") as f:
        for name in ("chi/Center of mass", "chi/FWHM", "mu/Center of mass", "mu/FWHM"):
            f.create_dataset(name, data=np.zeros((4, 8, 16), dtype="float64"))
    strain_path = tmp_path / "stacked_strain_volumes.h5"
    with h5py.File(strain_path, "w") as f:
        f.create_dataset("strain", data=np.zeros((4, 8, 16), dtype="float64"))

    est = estimate({"mosa_volume_file": str(mosa_path), "strain_volume_file": str(strain_path)})
    assert est.peak_bytes > 0, "fixture did not reach the priced return"
    assert est.confidence == "measured"


def test_slices_estimator_is_not_marked(tmp_path):
    from dfxm.stages.slices import estimate

    path = tmp_path / "mosa.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("chi/Center of mass", data=np.zeros((4, 8, 16), dtype="float32"))
    params = {"mosa_volume_file": str(path), **_SLICES_ALL_TOGGLES_OFF}
    params["include_mosa_com_chi"] = True
    est = estimate(params)
    assert est.peak_bytes > 0, "fixture did not reach the priced return"
    assert est.confidence == "measured"


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


def test_matched_prices_the_samy_padding_the_run_will_add(tmp_path):
    """`_apply_shift_single` builds an `(ny, nx + pads)` canvas per layer and the
    figure rasterises it. On the real STO2 dataset the pad is 1058 px on a
    2048-wide detector — a 52% inflation, and the single largest reason the
    first recalibration of this stage under-predicted the real run by 2.3x.
    """
    from dfxm.common import alignment as A
    from dfxm.stages import matched
    from dfxm.stages.matched import estimate

    ny = nx = 64
    px = 0.5
    # 40 px of total shift over 7 steps, one direction: pad_left = 40, right = 0.
    step = 40 * px / 1000.0 / 7
    flat = _raw_scans(tmp_path / "flat", n_scans=8, n_frames=4, ny=ny, nx=nx)
    spread = _raw_scans(tmp_path / "spread", n_scans=8, n_frames=4, ny=ny, nx=nx, samy_step_mm=step)
    base = {
        "rocking_pattern": "rock__*",
        "strain_pattern": "rock__*",
        "pixel_size_x_um": px,
        "samy_direction": -1,
    }
    samy = np.array([step * i for i in range(8)])
    pad = int(A.compute_pad_left(samy, px, -1)) + int(A.compute_pad_right(samy, px, -1))
    # Precondition: the fixture must actually pad, or this tests nothing.
    assert pad > 0, "fixture produced no padding"

    assert estimate({**base, "raw_root": flat}).peak_bytes == _matched_peak(
        n_frames=4, ny=ny, nx=nx
    )
    assert estimate({**base, "raw_root": spread}).peak_bytes == _matched_peak(
        n_frames=4, ny=ny, nx=nx, pad=pad
    )
    assert (
        estimate({**base, "raw_root": spread}).peak_bytes
        - estimate({**base, "raw_root": flat}).peak_bytes
        == matched.MATCHED_PADDED_BYTES_PER_ELEM * ny * pad
    )


def test_matched_drops_the_clim_pool_when_both_limits_are_given(tmp_path):
    """The ten-frame colour-limit pool only runs when `vmin`/`vmax` are blank.

    Measured as the auto-minus-fixed delta at five frame sizes with fourteen
    folders — enough to reach the ten-frame cap, which a four-folder fixture
    never does. That is the regime the first recalibration measured in, and why
    it under-charged this term.
    """
    from dfxm.stages import matched
    from dfxm.stages.matched import estimate

    root = _raw_scans(tmp_path / "raw", n_scans=12, n_frames=4, ny=32, nx=32)
    base = {"raw_root": root, "rocking_pattern": "rock__*", "strain_pattern": "rock__*"}
    auto = estimate(base).peak_bytes
    fixed = estimate({**base, "vmin": "0", "vmax": "1000"}).peak_bytes
    assert auto - fixed == matched.MATCHED_CLIM_POOL_BYTES_PER_ELEM * 32 * 32
    # One limit alone still leaves the pool running — run() needs the other.
    assert estimate({**base, "vmin": "0"}).peak_bytes == auto
    assert estimate({**base, "vmax": "1000"}).peak_bytes == auto


def test_matched_estimate_survives_unreadable_motors(tmp_path):
    """No samy dataset at all must mean "no padding", not an exception: a run
    whose motors are missing does no shifting either.
    """
    from dfxm.stages.matched import estimate

    root = tmp_path / "raw"
    scan = root / "rock__0"
    scan.mkdir(parents=True)
    with h5py.File(scan / "rock__0.h5", "w") as f:
        f.create_dataset("1.1/measurement/pco_ff", shape=(4, 8, 16), dtype="uint16")
    est = estimate(
        {"raw_root": str(root), "rocking_pattern": "rock__*", "strain_pattern": "rock__*"}
    )
    assert est.peak_bytes == _matched_peak(n_frames=4, ny=8, nx=16)


@pytest.mark.parametrize("pixel_size", ["", 0, 0.0, None, "junk"])
def test_matched_pad_is_zero_when_the_pixel_size_is_unusable(tmp_path, pixel_size):
    """The pads divide by the pixel size. A blank or zero one means the form is
    not ready — not that the canvas is unbounded.

    Asserted with warnings as errors, deliberately. Falling through to the
    division would give `inf`/`nan` offsets, and `int(nan)` raises `ValueError`
    straight into the fallback — so the estimate comes out the same either way
    and only the numpy RuntimeWarning distinguishes a checked pixel size from an
    accidental one. Without `simplefilter("error")` this test passes on an
    estimator that has no check at all.
    """
    import warnings

    from dfxm.stages.matched import estimate

    root = _raw_scans(tmp_path / "raw", n_scans=8, n_frames=4, ny=64, nx=64, samy_step_mm=0.01)
    params = {
        "raw_root": root,
        "rocking_pattern": "rock__*",
        "strain_pattern": "rock__*",
        "pixel_size_x_um": pixel_size,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        est = estimate(params)
    assert est.peak_bytes == _matched_peak(n_frames=4, ny=64, nx=64)


# -----------------------------------------------------------------------------
# What the 2026-08-26 review found, and what now pins it
# -----------------------------------------------------------------------------
def test_strain_canvas_follows_the_roi_the_render_will_crop_to(tmp_path):
    """`process_maps_file` hands `build_strain_map` the ROI-CROPPED map, and
    unlike the array term the crop does not merely shrink this one.

    The legacy and `figure_width` geometries fix the width and take the height
    from the physical aspect, so a narrow COLUMN crop makes the figure taller:
    at the STO2 pixel sizes, 1832 columns cropped to 200 takes the canvas from
    3.9 to 33.8 Mpx. Pricing the un-cropped layer under-predicts that 8x.
    """
    from dfxm.stages.strain import _plot_canvas_pixels

    p = {
        "pixel_size_x_um": 0.151733,
        "pixel_size_y_um": 0.387584,
        "roi": "",
        "plot_style": None,
    }
    shape = (1266, 1832)
    full = _plot_canvas_pixels(p, shape)
    tall = _plot_canvas_pixels({**p, "roi": "0,1266,800,1000"}, shape)
    wide = _plot_canvas_pixels({**p, "roi": "630,1266,105,1832"}, shape)
    # Precondition: the two crops must land on opposite sides of the uncropped
    # figure, or this passes on an estimator that ignores the ROI entirely.
    assert tall > full > wide, f"fixture is not two-sided: {wide} {full} {tall}"
    assert tall > 4 * full, "a narrow column crop must cost several times the full map"


def test_strain_canvas_is_not_computed_from_a_transposed_map(tmp_path):
    """Swapping the extents at the `strain_map_geometry` call site is invisible
    to a `styled > plain` assertion — the fixed-scale box has the same pixel
    count either way — so the canvas is pinned against an independently
    computed figure here.
    """
    from dfxm.common.plotting import canvas_pixels
    from dfxm.stages.strain import (
        DETREND_DIAG_DPI,
        DETREND_DIAG_FIGSIZE,
        STRAIN_MAP_DPI,
        _plot_canvas_pixels,
    )

    ny, nx, px, py = 1266, 1832, 0.151733, 0.387584
    aspect = (ny * py) / (nx * px)
    # Precondition: a transpose must be distinguishable, i.e. the map is not square.
    assert abs(aspect - 1.0 / aspect) > 0.1
    expected = canvas_pixels((7.0, 7.0 * aspect + 1.5), STRAIN_MAP_DPI) + canvas_pixels(
        DETREND_DIAG_FIGSIZE, DETREND_DIAG_DPI
    )
    got = _plot_canvas_pixels(
        {"pixel_size_x_um": px, "pixel_size_y_um": py, "roi": "", "plot_style": None}, (ny, nx)
    )
    assert got == expected


@pytest.mark.parametrize(
    "over",
    [
        {"pixel_size_x_um": ""},
        {"pixel_size_x_um": 0},
        {"pixel_size_y_um": float("inf")},
        {"pixel_size_y_um": float("nan")},
        {"plot_style": "junk"},
        {"roi": "1,2,3"},
        {"roi": "junk"},
    ],
)
def test_strain_canvas_never_raises_on_a_half_typed_form(over):
    """`_plot_canvas_pixels` runs on every GUI keystroke. An infinite pixel size
    reaches `int(round(inf))`, which raises OverflowError — not ValueError — so
    the except set is asserted against the real failure modes, not assumed.
    """
    from dfxm.stages.strain import _plot_canvas_pixels

    base = {"pixel_size_x_um": 0.151733, "pixel_size_y_um": 0.387584, "roi": "", "plot_style": None}
    px = _plot_canvas_pixels({**base, **over}, (1266, 1832))
    assert px > 0


def test_canvas_pixels_rounds_each_side_as_matplotlib_does():
    from dfxm.common.plotting import canvas_pixels

    assert canvas_pixels((7.0, 13.0), 200) == 1400 * 2600
    assert canvas_pixels((7.004, 13.0), 200) == 1401 * 2600  # 1400.8 -> 1401
    assert canvas_pixels((0.0, 13.0), 200) == 0
    assert canvas_pixels((-5.0, 13.0), 200) == 0  # clamped, never negative


def test_mosaicity_prices_the_widest_dataset_not_the_last_one(tmp_path):
    """`_DATASETS` order must not decide the model. A maps.h5 with float64 chi
    and float32 mu was priced at the mu itemsize — half the real cost.
    """
    from dfxm.stages import mosaicity
    from dfxm.stages.mosaicity import estimate

    root = tmp_path / "mixed"
    folder = root / "layer__1"
    folder.mkdir(parents=True)
    with h5py.File(folder / "maps.h5", "w") as f:
        # chi comes FIRST in _DATASETS and is the wide one; mu comes last.
        f.create_dataset(MOSA_PATHS["chi_com_path"], shape=(H, W), dtype="float64")
        f.create_dataset(MOSA_PATHS["mu_fwhm_path"], shape=(H, W), dtype="float32")
    est = estimate(_mosa_params(str(root)))
    assert est.peak_bytes == mosaicity.MOSAICITY_PROCESS_FLOOR_BYTES + (
        2 + mosaicity.MOSAICITY_WRITER_LAYERS
    ) * (H * W * 8)


def test_rocking_prices_the_aligned_volume_not_the_frames_it_read(tmp_path):
    """`run()` X-pads by the samy extremes and resamples Z; both inflate what
    accumulates. On real STO2 the aligned volume is (76, 700, 2891) against a
    1832-column read — a 1.58x inflation the model must carry.
    """
    from dfxm.common import alignment as A
    from dfxm.stages import rocking
    from dfxm.stages.rocking import estimate

    ny = nx = 64
    px = 0.5
    step = 40 * px / 1000.0 / 7  # 40 px of total shift over 7 steps
    flat = _raw_scans(tmp_path / "flat", n_scans=8, n_frames=3, ny=ny, nx=nx)
    spread = _raw_scans(tmp_path / "spread", n_scans=8, n_frames=3, ny=ny, nx=nx, samy_step_mm=step)
    base = {
        "rocking_pattern": "rock__*",
        "mosa_pattern": "rock__*",
        "pixel_size_x_um": px,
        "samy_direction": -1,
        "save_topview": False,
    }
    aligned = A.aligned_shape_for_params(
        {**rocking.STAGE.defaults(), **base, "raw_root": spread},
        (8, ny, nx),
        pattern_key="rocking_pattern",
    )
    # Precondition: the fixture must actually inflate, or this tests nothing.
    assert aligned is not None and aligned[2] > nx, f"no padding in the fixture: {aligned}"

    got = estimate({**base, "raw_root": spread}).peak_bytes
    assert got == _rocking_peak(
        n_scans=8, n_frames=3, ny=ny, nx=nx, volume_elems=aligned[0] * aligned[1] * aligned[2]
    )
    assert got > estimate({**base, "raw_root": flat}).peak_bytes


@pytest.mark.parametrize("dtype,itemsize", [("uint16", 2), ("float32", 4), ("float64", 8)])
def test_raw_stage_read_terms_follow_the_source_dtype(tmp_path, dtype, itemsize):
    """rocking's per-scan read is `itemsize + 4` and matched's median term is
    `itemsize + MEDIAN_WORKING_SET_PER_ELEMENT`; neither had a dtype test.
    """
    from dfxm.stages import matched
    from dfxm.stages.matched import estimate as matched_estimate
    from dfxm.stages.rocking import estimate as rocking_estimate

    root = _raw_scans(tmp_path / f"raw{dtype}", n_scans=2, n_frames=3, dtype=dtype)
    rock = rocking_estimate({"raw_root": root, "rocking_pattern": "rock__*", "save_topview": False})
    assert rock.peak_bytes == _rocking_peak(n_scans=2, n_frames=3, ny=8, nx=16, itemsize=itemsize)
    match = matched_estimate(
        {"raw_root": root, "rocking_pattern": "rock__*", "strain_pattern": "rock__*"}
    )
    # Precondition: below the block cap, or the itemsize would not appear at all.
    assert 3 * 8 * 16 * (itemsize + matched.MEDIAN_WORKING_SET_PER_ELEMENT) < (
        matched.MEDIAN_BLOCK_WORKING_SET_BYTES
    )
    assert match.peak_bytes == _matched_peak(n_frames=3, ny=8, nx=16, itemsize=itemsize)


@pytest.mark.parametrize("limit", ["-", "abc", "1e", ".", "-.", "1,2"])
def test_matched_survives_a_half_typed_colour_limit(tmp_path, limit):
    """`vmin`/`vmax` are free-text params and the auto-clim gate parses them, so
    the gate must absorb what a user types on the way to a number.

    `"-"` is the first keystroke of a negative limit. A limit that does not
    parse is one run() will not use either, so the pooled pass is what happens
    and charging it is the conservative direction as well as the correct one.
    """
    from dfxm.stages.matched import estimate

    root = _raw_scans(tmp_path / "raw", n_scans=2, n_frames=4, ny=32, nx=32)
    base = {"raw_root": root, "rocking_pattern": "rock__*", "strain_pattern": "rock__*"}
    priced = _matched_peak(n_frames=4, ny=32, nx=32)
    for over in ({"vmin": limit}, {"vmax": limit}, {"vmin": limit, "vmax": limit}):
        assert estimate({**base, **over}).peak_bytes == priced


def _mosa_scans(root, *, n_scans=2, n_frames=4, ny=8, nx=16, samy_mm=0.0):
    """Mosa folders whose samy anchors the alignment — `run()` uses mosa[0]."""
    import h5py

    for i in range(n_scans):
        scan = root / f"mosa__{i}"
        scan.mkdir(parents=True)
        with h5py.File(scan / f"mosa__{i}.h5", "w") as f:
            f.create_dataset("1.1/measurement/pco_ff", shape=(n_frames, ny, nx), dtype="uint16")
            f.create_dataset("1.1/instrument/positioners/samy", data=samy_mm)
            f.create_dataset("1.1/instrument/positioners/samz", data=0.0)
    return str(root)


def test_rocking_estimate_anchors_the_samy_shift_where_the_run_does(tmp_path):
    """`run()` anchors at `mosa_samy[0]` (rocking.py: samy_ref, samz_ref).
    Anchoring the ESTIMATE at the rocking glob's own first scan makes the pad —
    and with it the dominant 48 B/elem volume term — too SMALL, which is the
    under-prediction direction the recalibration exists to close."""
    from dfxm.common import alignment as A
    from dfxm.stages import rocking as RK

    root = tmp_path / "raw"
    step, n, ny, nx = 0.05, 4, 8, 16
    # Rocking scans sit ABOVE the mosaicity reference, so anchoring at the
    # rocking glob's first scan loses the whole mosa->rock offset.
    _raw_scans(root, n_scans=n, n_frames=4, ny=ny, nx=nx, samy_step_mm=step)
    for i in range(n):  # shift every rocking samy up by a full 10 steps
        import h5py

        with h5py.File(root / f"rock__{i}" / f"rock__{i}.h5", "r+") as f:
            del f["1.1/instrument/positioners/samy"]
            f.create_dataset("1.1/instrument/positioners/samy", data=step * (10 + i))
    _mosa_scans(root, samy_mm=0.0)

    p = {
        **RK.STAGE.defaults(),
        "raw_root": str(root),
        "rocking_pattern": "rock__*",
        "mosa_pattern": "mosa__*",
        "pixel_size_x_um": 1.0,
        "samy_direction": 1,
        "save_topview": False,
    }
    est = RK.estimate(p)

    samy = np.array([step * (10 + i) for i in range(n)])
    samz = np.array([0.002 * i for i in range(n)])
    anchored = A.aligned_extent(
        (n, ny, nx), samy, samz, scale_x=1.0, samy_direction=1, ref_samy=0.0, ref_samz=0.0
    )
    unanchored = A.aligned_extent((n, ny, nx), samy, samz, scale_x=1.0, samy_direction=1)
    # Precondition: the two anchorings really do disagree, or this is vacuous.
    assert anchored[2] > unanchored[2], (anchored, unanchored)

    expected = _rocking_peak(
        n_scans=n, n_frames=4, ny=ny, nx=nx, volume_elems=anchored[0] * anchored[1] * anchored[2]
    )
    assert est.peak_bytes == expected


def test_strain_out_of_range_roi_falls_back_instead_of_zeroing_the_canvas(_strain_defaults):
    """`min(ny, r1) - max(0, r0)` can go NEGATIVE, and `canvas_pixels` clamps a
    negative figure to 0 — silently deleting the largest plotting term instead
    of falling back to the legacy geometry. Nothing raises, so the cost line
    shows a number computed from a term that vanished."""
    from dfxm.stages.strain import _plot_canvas_pixels

    layer = (1266, 1832)
    good = _plot_canvas_pixels({**_strain_defaults, "roi": ""}, layer)
    bad = _plot_canvas_pixels({**_strain_defaults, "roi": "2000,3000,0,1832"}, layer)
    assert good > 0
    # The defect was the MAP canvas term vanishing entirely, leaving only the
    # detrend diagnostic. Assert the term survived — not which of the two
    # geometries is larger, which is not the property at stake.
    from dfxm.common.plotting import canvas_pixels
    from dfxm.stages.strain import DETREND_DIAG_DPI, DETREND_DIAG_FIGSIZE

    diag_only = canvas_pixels(DETREND_DIAG_FIGSIZE, DETREND_DIAG_DPI)
    assert bad > diag_only, (bad, diag_only)
    assert (bad - diag_only) > 0.5 * (good - diag_only), (good, bad, diag_only)
