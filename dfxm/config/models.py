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
    ``advanced`` params collapse into the form's Advanced expander under
    their ``group`` header; ``must_exist`` marks input paths the GUI
    verifies on disk before launching a run (never set it on outputs).
    """

    name: str
    type: ParamType
    label: str
    default: Any = None
    unit: str | None = None
    choices: tuple[Any, ...] | None = None
    help: str | None = None
    calibration: bool = False
    advanced: bool = False  # True -> rendered inside the collapsed Advanced expander
    group: str = ""  # themed section header inside Advanced (required when advanced)
    must_exist: bool = False  # input path/dir: GUI checks existence before a run
    roi_group: str = ""  # params sharing a roi_group are one ROI-picker target
    roi_axis: str = ""  # "" | "x" | "y" | "both" ("both" = one 4-int "r0,r1,c0,c1" field)
    roi_frame: str = ""  # "" | "detector" | "map" — the coordinate frame of a ROI param

    def __post_init__(self) -> None:
        if self.type is ParamType.ENUM and not self.choices:
            raise ValueError(f"enum param {self.name!r} needs a non-empty `choices`")
        if self.roi_axis and not self.roi_group:
            raise ValueError(f"roi param {self.name!r}: roi_axis set but roi_group is empty")
        if self.roi_axis not in ("", "x", "y", "both"):
            raise ValueError(f"roi param {self.name!r}: bad roi_axis {self.roi_axis!r}")
        if self.roi_frame not in ("", "detector", "map"):
            raise ValueError(f"roi param {self.name!r}: bad roi_frame {self.roi_frame!r}")

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
class CostEstimate:
    """What a stage run will cost, computed from HDF5 shapes alone.

    Produced by a stage's ``estimate(params)`` function, which is cheap enough
    to recompute on every form change: from the volume files it reads
    ``.shape``/``.dtype`` only, and **never a voxel**.

    The stages that price the alignment chain additionally read the **motor
    positions** — one scalar per raw scan folder — because the aligned array's
    extent depends on them and not on any shape in the volume file:
    ``apply_samy_shifts_to_volume`` widens image-X by the samy offsets and
    ``interpolate_to_uniform_z`` resamples Z onto a grid that exceeds the layer
    count when samz is irregular, and the two inflations multiply. Counting
    unpadded elements instead is what made those models under-predict, which is
    the dangerous direction. The read is memoised
    (:func:`~dfxm.common.raster.motor_positions_for_estimate`); a run never uses
    that cache.

    ``peak_bytes`` is the in-core high-water mark of the whole-volume strategy,
    including transient copies: a ``[:].astype(np.float64)`` on a float32 source
    holds both arrays at once and costs 3x the on-disk size, not 1x.
    ``chunkable`` is False for work that is irreducibly whole-array and must run
    disk-backed instead.

    ``chunk_span`` is ``(count, unit)`` naming what the chunking actually
    divides, for stages where that is **not** ``shape[0]`` layers.
    ``advice.plan_run`` otherwise reports "groups of N of ``shape[0]`` layers",
    which is right for the six stages that chunk their volume along Z and wrong
    for ``matched``, whose ``shape[0]`` is a count of scan folders while the
    thing it chunks is one scan's detector rows. Display only — nothing acts on
    the number — but a wrong unit in an advisory message is how advice stops
    being read.

    ``scratch_bytes`` is the **disk** a chunked run needs, as opposed to the
    memory ``peak_bytes`` prices. It is non-zero only where blocking the work
    forces a re-read cache: with ``center_method="median"`` the centring
    statistic is the one irreducibly whole-array step in the pipeline, so a
    chunked run spills its aligned blocks to scratch and reads them back.
    ``advice.plan_run`` checks it against ``profile.disk_free`` *before* the run
    starts — without it a machine short of disk discovers the problem halfway
    through a long run, which is precisely the failure this phase exists to
    prevent. Zero by default, so estimators that never spill are unaffected.
    """

    peak_bytes: int
    input_bytes: int
    shape: tuple[int, ...] | None
    chunkable: bool
    note: str | None = None
    chunk_span: tuple[int, str] | None = None
    scratch_bytes: int = 0


@dataclass(frozen=True)
class StageSpec:
    """A stage's identity plus the list of parameters it accepts."""

    name: str
    label: str
    description: str
    params: tuple[Param, ...]
    estimate: str | None = None  # "module:function" target, resolved lazily

    def estimator(self):
        """Resolve :attr:`estimate` to a callable, or None when unset.

        Kept as a string on the spec so importing the stage registry never
        drags in h5py/matplotlib; resolution happens only when a caller
        actually wants a prediction.
        """
        if self.estimate is None:
            return None
        from dfxm.stages.registry import resolve

        return resolve(self.estimate)

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
    pixel_size_x_um: float = 1.0  # detector pixel scale, X
    pixel_size_y_um: float = 1.0  # detector pixel scale, Y

    # --- regions of interest (frames + conversions: dfxm/common/roi.py) ------
    darfix_roi: str = ""  # darfix detector crop as darfix displays it: "x,y,w,h" origin+size
    analysis_roi_x: str = ""  # analysis window columns, map-frame "c0,c1" (blank = full)
    analysis_roi_y: str = ""  # analysis window rows, map-frame "r0,r1" (blank = full)

    # --- beamline HDF5 / motor paths (overridable constants) -----------------
    maps_filename: str = "maps.h5"  # darfix output filename inside each folder
    positioners_path: str = "instrument/positioners"  # relative to each entry
    detector_read_path: str = "instrument/pco_ff/image"  # raw detector frames
    detector_write_path: str = "measurement/pco_ff"  # concat output VDS location
    samy_key: str = "samy"  # sample-Y stage name under positioners
    samz_key: str = "samz"  # sample-Z stage name under positioners
    ccmth_com_path: str = "/entry/ccmth/Center of mass/Center of mass"

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
    Param(
        "name",
        ParamType.STR,
        "Preset name",
        help="Short name of the preset (used as the filename when saving).",
    ),
    Param(
        "description",
        ParamType.STR,
        "Description",
        help="One line describing the experiment/sample.",
    ),
    Param(
        "notes",
        ParamType.STR,
        "Notes",
        help=(
            "Free-text caveats shown in red in the GUI — e.g. calibration warnings "
            "for whoever loads this preset."
        ),
    ),
    Param(
        "raw_root",
        ParamType.DIR,
        "Raw data root",
        help=(
            "RAW_DATA root: the folder with the original beamline scan folders "
            "(input to concat, rocking and matched)."
        ),
    ),
    Param(
        "processed_root",
        ParamType.DIR,
        "Processed data root",
        help=(
            "PROCESSED_DATA root: where darfix wrote maps.h5 per layer "
            "(input to strain/mosaicity; the stacked volumes land here too)."
        ),
    ),
    Param(
        "folder_pattern",
        ParamType.STR,
        "Folder pattern",
        help="Glob for the concat/strain layer subfolders.",
    ),
    Param(
        "mosa_pattern",
        ParamType.STR,
        "Mosaicity pattern",
        help="Glob for the mosaicity layer subfolders (often *_mosa__*).",
    ),
    Param(
        "rocking_pattern",
        ParamType.STR,
        "Rocking pattern",
        help="Glob for the rocking scan subfolders (often *_rocking__*).",
    ),
    Param(
        "entry_suffix",
        ParamType.STR,
        "Entry suffix",
        help=(
            "BLISS entry filter for concat — only entries ending in this suffix "
            "are merged (e.g. '.1')."
        ),
    ),
    Param(
        "ccmth_ref_deg",
        ParamType.FLOAT,
        "ccmth reference",
        unit="deg",
        calibration=True,
        help=(
            "Reference Bragg angle of the unstrained lattice, in degrees. "
            "Strain is computed from deviations from this angle — a wrong value "
            "silently shifts every strain map. From the beamline alignment."
        ),
    ),
    Param(
        "pixel_size_x_um",
        ParamType.FLOAT,
        "Pixel size X",
        unit="µm",
        calibration=True,
        help=(
            "Detector pixel size along X in µm, from the beamline optics calibration. Pre-fills "
            "every stage's pixel size; several stages use it to convert the sample-Y motor shift "
            "(mm) into detector pixels, so a wrong value misaligns layers as well as scaling "
            "every reported distance."
        ),
    ),
    Param(
        "pixel_size_y_um",
        ParamType.FLOAT,
        "Pixel size Y",
        unit="µm",
        calibration=True,
        help=(
            "Detector pixel size along Y in µm, from the beamline optics calibration. Pre-fills "
            "every stage's pixel size; a wrong value skews the vertical physical scale of every "
            "map, volume, and export."
        ),
    ),
    Param(
        "darfix_roi",
        ParamType.STR,
        "Darfix ROI (origin+size)",
        help=(
            "The detector crop used in darfix, exactly as darfix's ROI widget shows it: "
            "'x,y,w,h' — origin then size (e.g. 105,230,1832,1266). Copy the four numbers "
            "verbatim, no conversion. Map pixel (0,0) sits at detector pixel (x,y); stages "
            "derive their detector-frame crops from this. Leave blank if darfix ran uncropped."
        ),
    ),
    Param(
        "analysis_roi_x",
        ParamType.STR,
        "Analysis window X (map px)",
        help=(
            "Columns of the darfix map to study, as 'c0,c1' start,end map pixels — relative "
            "to the darfix window, NOT absolute detector pixels. Pre-fills every stage's "
            "map-frame ROI X and (with the darfix ROI) rocking's detector crop. "
            "Blank = full width."
        ),
    ),
    Param(
        "analysis_roi_y",
        ParamType.STR,
        "Analysis window Y (map px)",
        help=(
            "Rows of the darfix map to study, as 'r0,r1' start,end map pixels — relative "
            "to the darfix window, NOT absolute detector pixels. Pre-fills every stage's "
            "map-frame ROI Y and (with the darfix ROI) rocking's detector crop. "
            "Blank = full height."
        ),
    ),
    Param(
        "maps_filename",
        ParamType.STR,
        "darfix maps filename",
        help="Filename darfix writes inside each layer folder (normally maps.h5).",
    ),
    Param(
        "positioners_path",
        ParamType.STR,
        "Positioners path",
        help="HDF5 path of the motor-position group inside each scan entry.",
    ),
    Param(
        "detector_read_path",
        ParamType.STR,
        "Detector read path",
        help="HDF5 path of the raw detector frames inside each scan entry.",
    ),
    Param(
        "detector_write_path",
        ParamType.STR,
        "Detector write path",
        help="HDF5 path where concat writes the merged detector data.",
    ),
    Param(
        "samy_key",
        ParamType.STR,
        "samy key",
        help="Name of the sample-Y translation motor under the positioners group.",
    ),
    Param(
        "samz_key",
        ParamType.STR,
        "samz key",
        help="Name of the sample-Z translation motor under the positioners group.",
    ),
    Param(
        "ccmth_com_path",
        ParamType.STR,
        "ccmth COM path",
        help="HDF5 path of the ccmth centre-of-mass dataset inside maps.h5.",
    ),
)

#: Names of the physically-meaningful calibration fields (for prominent flagging).
CALIBRATION_FIELDS: tuple[str, ...] = tuple(p.name for p in EXPERIMENT_SCHEMA if p.calibration)


def experiment_schema() -> tuple[Param, ...]:
    """Return the Experiment display schema (kept in sync with the dataclass)."""
    return EXPERIMENT_SCHEMA
