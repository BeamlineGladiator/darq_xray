"""Typed configuration models for the DFXM pipeline.

Two complementary pieces live here:

* :class:`Experiment` — the shared, preset-saved state (data roots, folder
  patterns, calibration angles, pixel scales, beamline HDF5/motor paths). One
  instance is loaded from a YAML preset and inherited by every stage.
* The :class:`Param` / :class:`StageSpec` schema framework — a tiny, Qt-free
  description of a stage's inputs. The GUI turns a schema into a form
  (enum -> dropdown, path -> file picker, number -> spin box); the CLI and
  tests read the same defaults. Keeping the schema declarative means the GUI
  never hard-codes a stage's fields.

Calibration fields (reference angles, pixel scales) are marked ``calibration=
True`` so the GUI can flag them visually — they are physically meaningful and
wrong values produce meaningless strain maps.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any


class ParamType(str, Enum):
    """Kind of a parameter, used by the GUI to pick an editor widget."""

    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    PATH = "path"  # an existing input file
    DIR = "dir"  # an existing input directory
    SAVE_PATH = "save_path"  # an output file path (need not exist yet)
    ENUM = "enum"  # one of a small finite set -> dropdown
    TEXT = "text"  # multi-line free text (e.g. JSON) -> text area


@dataclass(frozen=True)
class Param:
    """One declarative parameter in a stage schema.

    ``choices`` is required for :attr:`ParamType.ENUM` and renders as a
    dropdown. ``unit`` and ``help`` are advisory text for the form. Mark
    physically-meaningful calibration constants with ``calibration=True``.
    """

    name: str
    type: ParamType
    label: str
    default: Any = None
    unit: str | None = None
    choices: tuple[Any, ...] | None = None
    help: str | None = None
    calibration: bool = False

    def __post_init__(self) -> None:
        if self.type is ParamType.ENUM and not self.choices:
            raise ValueError(f"enum param {self.name!r} needs a non-empty `choices`")

    def coerce(self, value: Any) -> Any:
        """Convert a raw value (e.g. a string from a form field) to its type."""
        if value is None:
            return None
        if self.type is ParamType.INT:
            return int(value)
        if self.type is ParamType.FLOAT:
            return float(value)
        if self.type is ParamType.BOOL:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if self.type is ParamType.ENUM:
            if self.choices is not None and value not in self.choices:
                raise ValueError(f"{self.name}={value!r} not in {self.choices}")
            return value
        # STR, PATH, DIR, SAVE_PATH
        return str(value)


@dataclass(frozen=True)
class StageSpec:
    """A stage's identity plus the list of parameters it accepts."""

    name: str
    label: str
    description: str
    params: tuple[Param, ...]

    def defaults(self) -> dict[str, Any]:
        """Default value for every parameter, as a plain dict."""
        return {p.name: p.default for p in self.params}

    def get(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"no param named {name!r} in stage {self.name!r}")

    def coerce_all(self, values: dict[str, Any]) -> dict[str, Any]:
        """Coerce a dict of raw values, filling in defaults for missing keys."""
        out = self.defaults()
        for k, v in values.items():
            out[k] = self.get(k).coerce(v)
        return out


@dataclass
class Experiment:
    """Shared experiment state, loaded from a YAML preset.

    Field values below are placeholders; the shipped ``STO2_overnight`` preset
    fills in the real calibrated values. **Reference angles and pixel scales
    are experiment-specific** — see ``EXPERIMENT_SCHEMA`` for which fields are
    flagged as calibration.
    """

    name: str = "unnamed"
    description: str = ""
    notes: str = ""  # free-text, surfaced in the GUI (e.g. calibration caveats)

    # --- data roots ---------------------------------------------------------
    raw_root: str = ""  # RAW_DATA root (concat input)
    processed_root: str = ""  # PROCESSED_DATA root (darfix maps.h5 / strain input)
    folder_pattern: str = "*"  # glob for the per-layer subfolders (concat/strain)
    mosa_pattern: str = "*"  # glob for the mosaicity layer subfolders (often *_mosa__*)
    rocking_pattern: str = "*"  # glob for the rocking layer subfolders (often *_rocking__*)
    entry_suffix: str = ".1"  # BLISS entry filter for concat (e.g. 1.1, 2.1, ...)

    # --- calibration (physically meaningful — wrong values ruin the maps) ----
    ccmth_ref_deg: float = 0.0  # reference ccmth / monochromator Bragg angle
    mu_ref_deg: float = 0.0  # reference mu / sample Bragg angle (theta_s)
    pixel_size_x_um: float = 1.0  # detector pixel scale, X
    pixel_size_y_um: float = 1.0  # detector pixel scale, Y

    # --- beamline HDF5 / motor paths (overridable constants) -----------------
    maps_filename: str = "maps.h5"  # darfix output filename inside each folder
    positioners_path: str = "instrument/positioners"  # relative to each entry
    detector_read_path: str = "instrument/pco_ff/image"  # raw detector frames
    detector_write_path: str = "measurement/pco_ff"  # concat output VDS location
    samy_key: str = "samy"  # sample-Y stage name under positioners
    samz_key: str = "samz"  # sample-Z stage name under positioners
    ccmth_com_path: str = "/entry/ccmth/Center of mass/Center of mass"
    mu_com_path: str = "/entry/mu/Center of mass/Center of mass"

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for YAML serialisation (preserves field order)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experiment:
        """Build from a dict, ignoring unknown keys (with a warning)."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            import warnings

            warnings.warn(
                f"ignoring unknown experiment keys: {sorted(unknown)}",
                stacklevel=2,
            )
        return cls(**{k: v for k, v in data.items() if k in known})


# Declarative schema describing the Experiment fields for the GUI: labels,
# units, and which fields are calibration constants. The order here is the
# display order. A test asserts these names stay in sync with the dataclass.
EXPERIMENT_SCHEMA: tuple[Param, ...] = (
    Param("name", ParamType.STR, "Preset name"),
    Param("description", ParamType.STR, "Description"),
    Param("notes", ParamType.STR, "Notes"),
    Param("raw_root", ParamType.DIR, "Raw data root", help="RAW_DATA root (concat input)"),
    Param(
        "processed_root",
        ParamType.DIR,
        "Processed data root",
        help="PROCESSED_DATA root (darfix maps.h5 / strain input)",
    ),
    Param(
        "folder_pattern",
        ParamType.STR,
        "Folder pattern",
        help="glob for concat/strain layer subfolders",
    ),
    Param(
        "mosa_pattern",
        ParamType.STR,
        "Mosaicity pattern",
        help="glob for mosaicity layer subfolders",
    ),
    Param(
        "rocking_pattern",
        ParamType.STR,
        "Rocking pattern",
        help="glob for rocking layer subfolders",
    ),
    Param("entry_suffix", ParamType.STR, "Entry suffix", help="BLISS entry filter, e.g. .1"),
    Param("ccmth_ref_deg", ParamType.FLOAT, "ccmth reference", unit="deg", calibration=True),
    Param("mu_ref_deg", ParamType.FLOAT, "mu reference", unit="deg", calibration=True),
    Param("pixel_size_x_um", ParamType.FLOAT, "Pixel size X", unit="µm", calibration=True),
    Param("pixel_size_y_um", ParamType.FLOAT, "Pixel size Y", unit="µm", calibration=True),
    Param("maps_filename", ParamType.STR, "darfix maps filename"),
    Param("positioners_path", ParamType.STR, "Positioners path"),
    Param("detector_read_path", ParamType.STR, "Detector read path"),
    Param("detector_write_path", ParamType.STR, "Detector write path"),
    Param("samy_key", ParamType.STR, "samy key"),
    Param("samz_key", ParamType.STR, "samz key"),
    Param("ccmth_com_path", ParamType.STR, "ccmth COM path"),
    Param("mu_com_path", ParamType.STR, "mu COM path"),
)

#: Names of the physically-meaningful calibration fields (for prominent flagging).
CALIBRATION_FIELDS: tuple[str, ...] = tuple(p.name for p in EXPERIMENT_SCHEMA if p.calibration)


def experiment_schema() -> tuple[Param, ...]:
    """Return the Experiment display schema (kept in sync with the dataclass)."""
    return EXPERIMENT_SCHEMA
