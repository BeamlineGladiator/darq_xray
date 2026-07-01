"""Tests for dfxm.common.pixel_size."""

from __future__ import annotations

import math

import h5py
import pytest

from dfxm.common.errors import StageUserError
from dfxm.common.pixel_size import compute_pixel_size


def _write_scan(path, *, mainx, obx, ffsel, ffz, lenssel, entry="1.1", motors=None):
    """Write a minimal BLISS-style scan with only a positioners group."""
    with h5py.File(path, "w") as f:
        pos = f.create_group(f"{entry}/instrument/positioners")
        values = {
            "mainx": mainx,
            "obx": obx,
            "ffsel": ffsel,
            "ffz": ffz,
            "lenssel": lenssel,
        }
        if motors is not None:
            values = {k: v for k, v in values.items() if k in motors}
        for name, val in values.items():
            pos.create_dataset(name, data=val)
    return str(path)


def test_2x_condenser_in(tmp_path):
    p = _write_scan(
        tmp_path / "s.h5", mainx=5000.0, obx=273.0, ffsel=-60.0, ffz=2100.0, lenssel=0.0
    )
    res = compute_pixel_size(p)
    m = 5000.0 / 273.0 - 1.0
    two_theta = math.atan2(2100.0, 5000.0)
    assert res.objective == "2x"
    assert res.condenser_in is True
    assert res.magnification == pytest.approx(m)
    assert res.pixel_size_x_um == pytest.approx(3.25 / m)
    assert res.pixel_size_y_um == pytest.approx((3.25 / m) / math.sin(two_theta))
    assert res.two_theta_deg == pytest.approx(math.degrees(two_theta))


def test_10x_condenser_out(tmp_path):
    p = _write_scan(tmp_path / "s.h5", mainx=5000.0, obx=273.0, ffsel=0.0, ffz=2100.0, lenssel=3.0)
    res = compute_pixel_size(p)
    m = 5000.0 / 273.0 - 1.0
    assert res.objective == "10x"
    assert res.condenser_in is False
    assert res.pixel_size_x_um == pytest.approx(0.65 / m)
    # condenser out -> Y equals X (no sin(2theta) division)
    assert res.pixel_size_y_um == pytest.approx(0.65 / m)


def test_unrecognized_ffsel_raises(tmp_path):
    p = _write_scan(
        tmp_path / "s.h5", mainx=5000.0, obx=273.0, ffsel=-30.0, ffz=2100.0, lenssel=0.0
    )
    with pytest.raises(StageUserError):
        compute_pixel_size(p)


def test_missing_motor_raises(tmp_path):
    p = _write_scan(
        tmp_path / "s.h5",
        mainx=5000.0,
        obx=273.0,
        ffsel=-60.0,
        ffz=2100.0,
        lenssel=0.0,
        motors={"mainx", "obx", "ffsel", "ffz"},  # no lenssel
    )
    with pytest.raises(StageUserError):
        compute_pixel_size(p)


def test_no_matching_entry_raises(tmp_path):
    p = _write_scan(
        tmp_path / "s.h5",
        mainx=5000.0,
        obx=273.0,
        ffsel=-60.0,
        ffz=2100.0,
        lenssel=0.0,
        entry="1.2",  # does not end in the default ".1"
    )
    with pytest.raises(StageUserError):
        compute_pixel_size(p)


def test_zero_obx_raises(tmp_path):
    p = _write_scan(tmp_path / "s.h5", mainx=5000.0, obx=0.0, ffsel=-60.0, ffz=2100.0, lenssel=0.0)
    with pytest.raises(StageUserError):
        compute_pixel_size(p)
