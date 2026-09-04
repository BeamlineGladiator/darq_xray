"""Load, save, and discover experiment presets stored as YAML.

Presets live in the ``experiments/`` folder at the project root (one
``<name>.yaml`` per experiment) and ship pre-filled with the calibrated
constants for that beamtime. The GUI lists them in a dropdown; "Save as"
writes a new YAML here.

**This requires an editable install.** ``experiments/`` sits beside the
package rather than inside it and is deliberately not shipped as package
data, so :func:`experiments_dir` only resolves to real files when
``darq_xray`` is imported from a checkout (``pip install -e .``). After a
plain ``pip install .`` the directory lands inside ``site-packages`` and
does not exist, :func:`discover_experiments` returns ``{}``, and the app
starts with an empty preset dropdown. Pass *directory* explicitly to read
presets from anywhere else.

Comments in a hand-written preset (e.g. per-field unit notes) are
ignored on load and not regenerated on save — machine-relevant caveats should
also live in the :attr:`Experiment.notes` field so they survive a round-trip.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .models import Experiment


def experiments_dir() -> Path:
    """Default presets directory: ``<project root>/experiments``."""
    # darq_xray/config/presets.py -> project root is three parents up.
    return Path(__file__).resolve().parents[2] / "experiments"


def discover_experiments(directory: str | os.PathLike | None = None) -> dict[str, Path]:
    """Map preset name -> YAML path for every ``*.yaml`` in *directory*.

    The name is the file stem unless the YAML carries its own ``name`` key,
    in which case that wins. Sorted by name for stable dropdown ordering.
    """
    base = Path(directory) if directory is not None else experiments_dir()
    found: dict[str, Path] = {}
    if not base.is_dir():
        return found
    for path in sorted(base.glob("*.yaml")):
        name = path.stem
        try:
            with open(path, "r") as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, dict) and data.get("name"):
                name = str(data["name"])
        except (yaml.YAMLError, OSError):
            # Unreadable preset: fall back to the file stem rather than crash.
            pass
        found[name] = path
    return dict(sorted(found.items()))


def load_experiment(path: str | os.PathLike) -> Experiment:
    """Read a YAML preset into an :class:`Experiment`."""
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"preset {os.fspath(path)!r} is not a YAML mapping")
    return Experiment.from_dict(data)


def save_experiment(experiment: Experiment, path: str | os.PathLike) -> Path:
    """Write *experiment* to *path* as YAML; returns the path written.

    Uses ``sort_keys=False`` so the dataclass field order (and therefore the
    grouping) is preserved. Comments are not emitted.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        yaml.safe_dump(experiment.to_dict(), fh, sort_keys=False, allow_unicode=True)
    return out


def load_experiment_by_name(name: str, directory: str | os.PathLike | None = None) -> Experiment:
    """Discover presets in *directory* and load the one called *name*."""
    presets = discover_experiments(directory)
    if name not in presets:
        raise KeyError(f"no preset named {name!r} in {directory or experiments_dir()}")
    return load_experiment(presets[name])
