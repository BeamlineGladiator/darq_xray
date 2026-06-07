"""Glue between the Qt-free core and the GUI.

* ``STAGE_ORDER`` / ``STAGE_SPECS`` — the stages the GUI exposes and their
  parameter schemas.
* ``experiment_overrides`` — how an :class:`~dfxm.config.models.Experiment`
  pre-fills each stage's form (shared paths/angles entered once), and how an
  upstream stage's output auto-fills the next stage's input.

Run targets themselves come from :data:`dfxm.stages.registry.STAGE_TARGETS`.
"""

from __future__ import annotations

import os

from dfxm.config.models import Experiment, StageSpec
from dfxm.stages import concat, mosaicity, paraview, rocking, strain, visualize

# Display order in the navigation panel (pipeline order; darfix runs between
# concat and strain/mosaicity, outside the app).
STAGE_ORDER: tuple[str, ...] = (
    "concat",
    "strain",
    "mosaicity",
    "rocking",
    "visualize",
    "paraview",
)

STAGE_SPECS: dict[str, StageSpec] = {
    "concat": concat.STAGE,
    "strain": strain.STAGE,
    "mosaicity": mosaicity.STAGE,
    "rocking": rocking.STAGE,
    "visualize": visualize.STAGE,
    "paraview": paraview.STAGE,
}

# Default stacked-output filenames (kept in sync with each stage's defaults) so
# downstream stages can auto-fill their inputs.
_STRAIN_VOLUME = "stacked_strain_volumes.h5"
_MOSA_VOLUME = "stacked_volumes.h5"


def experiment_overrides(stage_name: str, exp: Experiment) -> dict:
    """Experiment-derived defaults that pre-fill *stage_name*'s form."""
    if stage_name == "concat":
        return dict(
            root_folder=exp.raw_root,
            folder_pattern=exp.folder_pattern,
            entry_suffix=exp.entry_suffix,
            detector_read_path=exp.detector_read_path,
            detector_write_path=exp.detector_write_path,
            positioners_path=exp.positioners_path,
        )
    if stage_name == "strain":
        return dict(
            root_folder=exp.processed_root,
            folder_pattern=exp.folder_pattern,
            maps_filename=exp.maps_filename,
            ccmth_com_path=exp.ccmth_com_path,
            mu_com_path=exp.mu_com_path,
            ccmth_ref_deg=exp.ccmth_ref_deg,
            mu_ref_deg=exp.mu_ref_deg,
            pixel_size_x_um=exp.pixel_size_x_um,
            pixel_size_y_um=exp.pixel_size_y_um,
        )
    if stage_name == "mosaicity":
        return dict(
            root_folder=exp.processed_root,
            folder_pattern=exp.mosa_pattern,
            maps_filename=exp.maps_filename,
        )
    if stage_name == "visualize":
        proc = exp.processed_root.rstrip("/")
        return dict(
            mosa_volume_file=os.path.join(proc, _MOSA_VOLUME) if proc else "",
            strain_volume_file=os.path.join(proc, _STRAIN_VOLUME) if proc else "",
            raw_root=exp.raw_root,
            mosa_pattern=exp.mosa_pattern,
            strain_pattern=exp.folder_pattern,
            # motor positions live under the first BLISS scan entry (1.1)
            samy_path=f"1.1/{exp.positioners_path}/{exp.samy_key}",
            samz_path=f"1.1/{exp.positioners_path}/{exp.samz_key}",
            pixel_size_x_um=exp.pixel_size_x_um,
            pixel_size_y_um=exp.pixel_size_y_um,
        )
    if stage_name == "rocking":
        return dict(
            raw_root=exp.raw_root,
            rocking_pattern=exp.rocking_pattern,
            mosa_pattern=exp.mosa_pattern,
            strain_pattern=exp.folder_pattern,
            samy_path=f"1.1/{exp.positioners_path}/{exp.samy_key}",
            samz_path=f"1.1/{exp.positioners_path}/{exp.samz_key}",
            # raw rocking frames live at the measurement soft-link, under 1.1
            detector_path=f"1.1/{exp.detector_write_path}",
            pixel_size_x_um=exp.pixel_size_x_um,
            pixel_size_y_um=exp.pixel_size_y_um,
        )
    if stage_name == "paraview":
        proc = exp.processed_root.rstrip("/")
        return dict(
            mosa_volume_file=os.path.join(proc, _MOSA_VOLUME) if proc else "",
            strain_volume_file=os.path.join(proc, _STRAIN_VOLUME) if proc else "",
            raw_root=exp.raw_root,
            mosa_pattern=exp.mosa_pattern,
            strain_pattern=exp.folder_pattern,
            samy_path=f"1.1/{exp.positioners_path}/{exp.samy_key}",
            samz_path=f"1.1/{exp.positioners_path}/{exp.samz_key}",
            pixel_size_x_um=exp.pixel_size_x_um,
            pixel_size_y_um=exp.pixel_size_y_um,
        )
    return {}
