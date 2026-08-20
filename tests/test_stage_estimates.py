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


def test_slices_is_not_chunkable_and_doubles_for_alignment(tmp_path):
    """astype(float64) plus an aligned copy — three arrays' worth at the peak."""
    from dfxm.stages.slices import estimate

    path = tmp_path / "mosa.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=np.zeros((4, 8, 16), dtype="float32"))
    est = estimate({"mosa_volume_file": str(path)})
    n = 4 * 8 * 16
    assert est.chunkable is False
    assert est.input_bytes == n * 4
    assert est.peak_bytes == n * 4 + 2 * n * 8


def test_matched_is_not_chunkable(tmp_path):
    """An exact median needs the whole stack — bucket 3, disk-backed."""
    from dfxm.stages.matched import estimate

    root = tmp_path / "raw"
    scan = root / "rock__1"
    scan.mkdir(parents=True)
    with h5py.File(scan / "rock__1.h5", "w") as f:
        f.create_dataset("1.1/measurement/pco_ff", data=np.zeros((6, 8, 16), dtype="uint16"))
    est = estimate({"raw_root": str(root), "rocking_pattern": "rock__*"})
    assert est.chunkable is False


def test_rocking_peak_has_no_conversion_overhead(tmp_path):
    """rocking.py:985 is a bare `dataset[:]` — no float64 copy to account for."""
    from dfxm.stages.rocking import estimate

    root = tmp_path / "raw"
    for i in range(2):
        scan = root / f"rock__{i}"
        scan.mkdir(parents=True)
        with h5py.File(scan / f"rock__{i}.h5", "w") as f:
            f.create_dataset("1.1/measurement/pco_ff", data=np.zeros((3, 8, 16), dtype="uint16"))
    est = estimate({"raw_root": str(root), "rocking_pattern": "rock__*"})
    assert est.peak_bytes == est.input_bytes
    assert est.input_bytes == 2 * 3 * 8 * 16 * 2  # 2 scans x 3 frames x uint16
