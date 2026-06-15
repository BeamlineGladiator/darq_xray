"""Stages raise StageUserError (with actionable hints) on bad user inputs."""

import pytest

from dfxm.common.errors import StageUserError
from dfxm.stages import concat, mosaicity, profiles, strain


def test_concat_single_requires_input_folder():
    with pytest.raises(StageUserError) as exc_info:
        concat.run({"mode": "single", "input_folder": ""})
    assert exc_info.value.hint


def test_strain_batch_no_matching_folders(tmp_path):
    with pytest.raises(StageUserError) as exc_info:
        strain.run({"mode": "batch", "root_folder": str(tmp_path), "folder_pattern": "zzz*"})
    assert "zzz*" in str(exc_info.value)
    assert "Folder pattern" in exc_info.value.hint


def test_mosaicity_batch_requires_root_folder():
    with pytest.raises(StageUserError) as exc_info:
        mosaicity.run({"mode": "batch", "root_folder": ""})
    assert exc_info.value.hint


def test_profiles_missing_slices_file(tmp_path):
    with pytest.raises(StageUserError) as exc_info:
        profiles.run({"consolidated_h5": str(tmp_path / "nope.h5")})
    assert "slices" in exc_info.value.hint
