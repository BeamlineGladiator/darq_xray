"""Glue between the Qt-free core and the GUI.

* ``STAGE_SPECS`` — the parameter schema for each stage the GUI exposes.
* ``experiment_overrides`` — how an :class:`~dfxm.config.models.Experiment`
  pre-fills a stage's form (so shared paths/angles are entered once).

Run targets themselves come from :data:`dfxm.stages.registry.STAGE_TARGETS`.
New stages are added here (spec) and in the registry (target) as they land.
"""

from __future__ import annotations

from dfxm.config.models import Experiment, StageSpec
from dfxm.stages import concat

# Display order of the stages in the navigation panel.
STAGE_ORDER: tuple[str, ...] = ("concat",)

STAGE_SPECS: dict[str, StageSpec] = {
    "concat": concat.STAGE,
}


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
    return {}
