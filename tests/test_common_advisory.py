"""Composition of profile + estimate + plan into one Advisory (Qt-free)."""

from __future__ import annotations

import os

from dfxm.common.advisory import disk_probe_dir
from dfxm.config.models import Param, ParamType, StageSpec

_SPEC = StageSpec(
    name="demo",
    label="Demo",
    description="",
    params=(
        Param("mosa_volume_file", ParamType.PATH, "Volume", must_exist=True),
        Param("root_folder", ParamType.DIR, "Root", must_exist=True),
        Param("output_dir", ParamType.DIR, "Out"),
    ),
)


def test_output_dir_wins_when_set(tmp_path):
    out = str(tmp_path / "out")
    assert disk_probe_dir(_SPEC, {"output_dir": out, "root_folder": "/elsewhere"}) == out


def test_falls_back_to_the_input_files_directory(tmp_path):
    """The branch that matters: an unset output_dir must NOT land on cwd while
    the data lives on another filesystem."""
    vol = tmp_path / "data" / "volumes.h5"
    vol.parent.mkdir(parents=True)
    vol.write_bytes(b"")
    got = disk_probe_dir(_SPEC, {"output_dir": "", "mosa_volume_file": str(vol)})
    assert got == str(vol.parent)
    assert got != os.getcwd()  # precondition: the fixture really is elsewhere


def test_falls_back_to_an_input_directory_unchanged(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert disk_probe_dir(_SPEC, {"output_dir": "", "root_folder": str(root)}) == str(root)


def test_falls_back_to_cwd_when_nothing_is_filled_in():
    assert disk_probe_dir(_SPEC, {}) == os.getcwd()


def test_ignores_params_that_are_not_inputs(tmp_path):
    """A non-must_exist path must never be chosen as the probe target."""
    spec = StageSpec(
        name="demo",
        label="Demo",
        description="",
        params=(Param("some_output", ParamType.SAVE_PATH, "Out"),),
    )
    assert disk_probe_dir(spec, {"some_output": str(tmp_path / "x.h5")}) == os.getcwd()
