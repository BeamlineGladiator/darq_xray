"""ParaView export stage — aligned mosaicity/strain volumes to PVTI.

Faithful port of ``export_aligned_volumes_to_paraview_v6_pvti.py``. It runs the
SAME alignment pipeline as the visualize stage (reusing
:mod:`dfxm.common.alignment`, so the PVTI is voxel-identical / co-registered),
then writes a partitioned VTK XML ImageData dataset: a ``.pvti`` master plus
per-piece ``.vti`` files split along Z, with a ``valid_mask`` field and NaN
sentinels so ParaView's GPU volume mapper never sees NaN.

``vtk`` is imported lazily inside the writer functions, so importing this module
(for its parameter schema) does not drag VTK into GUI startup.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import h5py
import numpy as np

from ..common import alignment as A
from ..common.h5io import sum_dataset_bytes
from ..common.raster import extract_motor_positions
from ..common.sort import find_matching_folders
from ..config.models import CostEstimate, Param, ParamType, StageSpec

ProgressFn = Callable[[float, str], None]

SAVE_DTYPE = np.float32  # float32 is plenty for visualisation

# What a child running this stage costs resident before it touches a voxel:
# interpreter, numpy, h5py and VTK. `tracemalloc` cannot see any of it, so it is
# what :func:`~dfxm.common.advice.working_set_budget_bytes` must take off the
# machine's headroom before converting the rest into an allocation budget.
#
# Measured at 229 MB (`tests/peak_rss.py` on a 4x8x8 four-field export, where
# the data is negligible), reproducibly to +/-0.1 MB. 300 MB is the figure: the
# extra is deliberate slack, not rounding. The additive RSS model is not an
# envelope — at a fixed traced peak, measured RSS varies by ~45 MB with the
# blocking alone — and this is the term with room to absorb that, since
# over-stating the floor only shrinks the budget while under-stating it invites
# an OOM. It also leaves room for a heavier VTK build than this one.
#
# Per stage by construction: it is set by what this module imports, and a stage
# that never touches VTK sits hundreds of MB below it. Pinned against a live
# measurement by `test_rss_floor_covers_the_measured_process_image`, so it fails
# loudly on the first machine or VTK build where it stops travelling rather than
# silently over-stating the budget.
RSS_FLOOR_BYTES = 300 * 1024 * 1024


def _noop(_frac: float, _msg: str) -> None:
    pass


STAGE = StageSpec(
    name="paraview",
    label="ParaView export (PVTI)",
    description=(
        "Aligns the mosaicity/strain volumes and exports them as partitioned .pvti datasets "
        "(with a validity mask) for 3-D volume rendering in ParaView, outside this app."
    ),
    params=(
        Param(
            "mosa_volume_file",
            ParamType.PATH,
            "Mosaicity volume",
            must_exist=True,
            help=(
                "The stacked mosaicity volume (stacked_volumes.h5) from the mosaicity stage. "
                "Leave blank to skip the mosaicity export."
            ),
        ),
        Param(
            "strain_volume_file",
            ParamType.PATH,
            "Strain volume",
            must_exist=True,
            help=(
                "The stacked strain volume (stacked_strain_volumes.h5) from the strain stage. "
                "Leave blank to skip the strain export."
            ),
        ),
        Param(
            "raw_root",
            ParamType.DIR,
            "Raw data root",
            must_exist=True,
            help=(
                "RAW_DATA root with the original scan folders — the samy/samz motor positions "
                "read from there drive the alignment."
            ),
        ),
        Param(
            "mosa_pattern",
            ParamType.STR,
            "Mosaicity raw pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help=(
                "Glob matching the raw mosaicity scan folders, used to read their samy/samz "
                "positions."
            ),
        ),
        Param(
            "strain_pattern",
            ParamType.STR,
            "Strain raw pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help=(
                "Glob matching the raw strain scan folders, used to read their samy/samz positions."
            ),
        ),
        Param(
            "samy_path",
            ParamType.STR,
            "samy path",
            default="1.1/instrument/positioners/samy",
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path to the sample-Y motor position inside each scan file (under the first "
                "BLISS entry). Only change for a different beamline file layout."
            ),
        ),
        Param(
            "samz_path",
            ParamType.STR,
            "samz path",
            default="1.1/instrument/positioners/samz",
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path to the sample-Z motor position inside each scan file (under the first "
                "BLISS entry). Only change for a different beamline file layout."
            ),
        ),
        Param(
            "pixel_size_x_um",
            ParamType.FLOAT,
            "Pixel size X",
            unit="µm",
            default=0.152,
            calibration=True,
            advanced=True,
            group="Calibration",
            help=(
                "Physical size of one detector pixel along X, in µm, from the beamline optics "
                "calibration. This is what converts the sample-Y motor shift (mm) into detector "
                "pixels during alignment, so a wrong value misaligns layers along X as well as "
                "scaling every exported voxel."
            ),
        ),
        Param(
            "pixel_size_y_um",
            ParamType.FLOAT,
            "Pixel size Y",
            unit="µm",
            default=0.385,
            calibration=True,
            advanced=True,
            group="Calibration",
            help=(
                "Physical size of one detector pixel along Y, in µm, from the beamline optics "
                "calibration. A wrong value skews the vertical voxel spacing of the export."
            ),
        ),
        Param(
            "samy_direction",
            ParamType.INT,
            "samy direction",
            default=-1,
            advanced=True,
            group="Alignment",
            help=(
                "Sign (+1 or −1) relating the samy motor direction to detector X. If features "
                "visibly march the wrong way between layers, flip the sign."
            ),
        ),
        Param(
            "roi_x",
            ParamType.STR,
            "Map ROI X",
            default="",
            roi_group="crop",
            roi_axis="x",
            roi_frame="map",
            help=(
                "Crop along map X as 'c0,c1' map pixels — columns of the darfix map, relative "
                "to the darfix window, NOT absolute detector pixels (blank = full width). "
                "Pre-filled from the experiment's analysis window. All volumes must share the "
                "same crop to stay co-registered."
            ),
        ),
        Param(
            "roi_y",
            ParamType.STR,
            "Map ROI Y",
            default="",
            roi_group="crop",
            roi_axis="y",
            roi_frame="map",
            help=(
                "Crop along map Y as 'r0,r1' map pixels — rows of the darfix map, relative to "
                "the darfix window, NOT absolute detector pixels (blank = full height). "
                "Pre-filled from the experiment's analysis window. All volumes must share the "
                "same crop to stay co-registered."
            ),
        ),
        Param(
            "center_method",
            ParamType.ENUM,
            "Centre method",
            default="mean",
            choices=("mean", "median"),
            advanced=True,
            group="Alignment",
            help=(
                "Statistic used to centre the misorientation values before export: mean or median."
            ),
        ),
        Param(
            "center_mosa_com",
            ParamType.BOOL,
            "Centre mosa CoM",
            default=True,
            advanced=True,
            group="Alignment",
            help=(
                "Subtract the centre statistic from the χ/μ CoM volumes so misorientation is "
                "relative to the bulk orientation."
            ),
        ),
        Param(
            "center_strain",
            ParamType.BOOL,
            "Centre strain",
            default=False,
            advanced=True,
            group="Alignment",
            help=(
                "Also centre the strain volume (usually off — strain is already relative to the "
                "reference angle)."
            ),
        ),
        Param(
            "abs_mosa_fwhm",
            ParamType.BOOL,
            "abs() FWHM",
            default=True,
            advanced=True,
            group="Alignment",
            help="Export FWHM as absolute values (darfix fits can produce negative widths).",
        ),
        Param(
            "anchor_origin_to_reference",
            ParamType.BOOL,
            "Anchor origin",
            default=False,
            advanced=True,
            group="Alignment",
            help=(
                "Place the world origin in the raw-detector frame shared with the rocking volume, "
                "so everything co-registers in ParaView."
            ),
        ),
        Param(
            "mosa_darfix_origin_xy",
            ParamType.STR,
            "Mosa darfix origin",
            default="105,230",
            roi_frame="detector",
            advanced=True,
            group="Alignment",
            help=(
                "Absolute detector pixels 'x,y' — the darfix crop origin used for the mosaicity "
                "maps, exactly as darfix's ROI widget shows it (copy verbatim, no conversion). "
                "Used only when 'Anchor origin' is on, to place the world origin in the shared "
                "raw-detector frame."
            ),
        ),
        Param(
            "strain_darfix_origin_xy",
            ParamType.STR,
            "Strain darfix origin",
            default="105,230",
            roi_frame="detector",
            advanced=True,
            group="Alignment",
            help=(
                "Absolute detector pixels 'x,y' — the darfix crop origin used for the strain "
                "maps, exactly as darfix's ROI widget shows it (copy verbatim, no conversion). "
                "Used only when 'Anchor origin' is on, to place the world origin in the shared "
                "raw-detector frame."
            ),
        ),
        Param(
            "num_pieces_z",
            ParamType.INT,
            "Z pieces",
            default=16,
            advanced=True,
            group="Export",
            help=(
                "Number of Z chunks the dataset is split into — match the MPI rank count of your "
                "pvserver for parallel rendering."
            ),
        ),
        Param(
            "piece_compression",
            ParamType.BOOL,
            "Compress pieces",
            default=False,
            advanced=True,
            group="Export",
            help="Compress the .vti pieces (smaller files, slower write).",
        ),
        Param(
            "replace_nan",
            ParamType.BOOL,
            "Replace NaN",
            default=True,
            advanced=True,
            group="Export",
            help="Replace NaN padding with a sentinel value so ParaView's volume renderer behaves.",
        ),
        Param(
            "write_valid_mask",
            ParamType.BOOL,
            "Write valid_mask",
            default=True,
            advanced=True,
            group="Export",
            help="Write a 0/1 valid_mask field — threshold on it in ParaView to hide the padding.",
        ),
        Param(
            "export_mosaicity",
            ParamType.BOOL,
            "Export mosaicity",
            default=True,
            advanced=True,
            group="Export",
            help="Export the mosaicity (χ/μ) volumes.",
        ),
        Param(
            "export_strain",
            ParamType.BOOL,
            "Export strain",
            default=True,
            advanced=True,
            group="Export",
            help="Export the strain volume.",
        ),
        Param(
            "output_dir",
            ParamType.DIR,
            "Output dir",
            help="Where the .pvti files and their piece folders are written.",
        ),
    ),
    estimate="dfxm.stages.paraview:estimate",
)


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------
@dataclass
class ExportInfo:
    name: str
    pvti_path: str
    dimensions_xyz: tuple[int, int, int]
    spacing_um_xyz: tuple[float, float, float]
    origin_um_xyz: tuple[float, float, float]
    n_pieces: int
    fields: list[str]


@dataclass
class ParaviewResult:
    output_dir: str = ""
    info_path: str | None = None
    exports: list[ExportInfo] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    # Advisories about HOW the run went, not what it skipped: a Z-piece count
    # too low for this machine's memory, or a median centring that had to
    # re-read because the scratch disk was full. Surfaced in the run log, the
    # Results summary and `export_info.txt`.
    notes: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# PVTI writer (faithful port; vtk imported lazily)
# -----------------------------------------------------------------------------
_NUMPY_TO_VTK_XML_TYPE = {
    np.float32: "Float32",
    np.float64: "Float64",
    np.int8: "Int8",
    np.uint8: "UInt8",
    np.int16: "Int16",
    np.uint16: "UInt16",
    np.int32: "Int32",
    np.uint32: "UInt32",
    np.int64: "Int64",
    np.uint64: "UInt64",
}


def _numpy_to_vtk_type_str(dtype) -> str:
    d = np.dtype(dtype)
    for np_t, vtk_t in _NUMPY_TO_VTK_XML_TYPE.items():
        if d == np.dtype(np_t):
            return vtk_t
    raise ValueError(f"Unsupported dtype for VTK XML: {dtype}")


def compute_piece_extents_z(nz: int, n_pieces: int) -> list:
    """Split Z range [0, nz-1] into n_pieces, adjacent pieces sharing one Z index."""
    n_pieces = max(1, min(int(n_pieces), max(1, nz - 1)))
    total_gaps = nz - 1
    gaps_per_piece = total_gaps // n_pieces
    extra_gaps = total_gaps % n_pieces
    extents = []
    start = 0
    for k in range(n_pieces):
        g = gaps_per_piece + (1 if k < extra_gaps else 0)
        end = start + g
        extents.append((int(start), int(end)))
        start = end
    return extents


def write_piece_vti(
    field_arrays: dict,
    piece_extent_z,
    whole_dims_xyz,
    spacing_xyz,
    origin_xyz,
    out_path,
    compression=False,
) -> None:
    """Write a single .vti piece with an explicit Z sub-extent."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    nx, ny, _ = whole_dims_xyz
    z0, z1 = piece_extent_z
    img = vtk.vtkImageData()
    img.SetExtent(0, nx - 1, 0, ny - 1, int(z0), int(z1))
    img.SetSpacing(*(float(s) for s in spacing_xyz))
    img.SetOrigin(*(float(o) for o in origin_xyz))

    first_name = None
    for name, piece_vol in field_arrays.items():
        flat = np.ascontiguousarray(piece_vol, dtype=SAVE_DTYPE).ravel(order="C")
        varr = numpy_to_vtk(flat, deep=True)
        varr.SetName(name)
        img.GetPointData().AddArray(varr)
        if first_name is None:
            first_name = name
    if first_name is not None:
        img.GetPointData().SetActiveScalars(first_name)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(img)
    writer.SetDataModeToBinary()
    if compression:
        writer.SetCompressorTypeToZLib()
    else:
        writer.SetCompressorTypeToNone()
    writer.Write()


def write_pvti_master(
    pvti_path, whole_dims_xyz, spacing_xyz, origin_xyz, piece_specs, field_specs
) -> None:
    """Write the .pvti master XML that references every piece."""
    nx, ny, nz = whole_dims_xyz
    default_scalar = field_specs[0][0] if field_specs else ""
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="PImageData" version="0.1" byte_order="LittleEndian">',
        (
            f'  <PImageData WholeExtent="0 {nx - 1} 0 {ny - 1} 0 {nz - 1}" GhostLevel="0" '
            f'Origin="{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}" '
            f'Spacing="{spacing_xyz[0]} {spacing_xyz[1]} {spacing_xyz[2]}">'
        ),
    ]
    ppdata_attr = f' Scalars="{default_scalar}"' if default_scalar else ""
    lines.append(f"    <PPointData{ppdata_attr}>")
    for name, dtype_str in field_specs:
        lines.append(f'      <PDataArray type="{dtype_str}" Name="{name}"/>')
    lines.append("    </PPointData>")
    for source_rel, (z0, z1) in piece_specs:
        lines.append(f'    <Piece Extent="0 {nx - 1} 0 {ny - 1} {z0} {z1}" Source="{source_rel}"/>')
    lines.append("  </PImageData>")
    lines.append("</VTKFile>")
    with open(pvti_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _sentinel_for(global_min: float, global_max: float) -> float:
    """The NaN replacement value, from the finite range of every field together.

    One definition for both writers: an in-core caller passes the range it read
    off the arrays, a streaming caller the range :func:`volumeio.stream_minmax`
    accumulated. ``global_min`` non-finite means no field held a finite voxel.
    """
    if not np.isfinite(global_min):
        return -1e30
    return global_min - 1000.0 * max(global_max - global_min, 1.0)


def _write_partitioned_vti(
    piece_fields,
    field_names: list,
    dims_zyx: tuple,
    spacing: tuple,
    output_path_pvti: str,
    *,
    origin: tuple,
    n_pieces: int,
    compression: bool,
    replace_nan: bool,
    write_valid_mask: bool,
    sentinel: float | None,
    nan_fraction_overall: float,
) -> dict:
    """Write the pieces and the ``.pvti`` manifest, and describe what was written.

    The single place that decides piece extents, file names, the manifest and
    the returned dict, so the in-core and streaming writers cannot drift apart
    in any of them — only in how they produce a piece's arrays.

    *piece_fields* is a callable ``(z0, z1) -> {name: array}`` returning the
    already-cleaned fields for the **inclusive** Z extent ``[z0, z1]``, sized to
    the piece. *field_names* is what those dicts will be keyed by, in order,
    needed up front for the manifest.
    """
    if not output_path_pvti.endswith(".pvti"):
        raise ValueError(f"output_path_pvti must end with .pvti: {output_path_pvti}")
    nz, ny, nx = dims_zyx

    extents = compute_piece_extents_z(nz, n_pieces)
    out_dir = os.path.dirname(os.path.abspath(output_path_pvti))
    base_no_ext = os.path.splitext(os.path.basename(output_path_pvti))[0]
    pieces_subdir_name = f"{base_no_ext}_pieces"
    pieces_dir = os.path.join(out_dir, pieces_subdir_name)
    os.makedirs(pieces_dir, exist_ok=True)

    whole_dims = (nx, ny, nz)
    spacing_tuple = tuple(float(s) for s in spacing)
    origin_tuple = tuple(float(o) for o in origin)
    n_pad = max(3, len(str(len(extents) - 1)))
    piece_specs = []
    total_size_bytes = 0

    for k, (z0, z1) in enumerate(extents):
        piece_basename = f"{base_no_ext}_piece_{k:0{n_pad}d}.vti"
        piece_path = os.path.join(pieces_dir, piece_basename)
        piece_rel = f"{pieces_subdir_name}/{piece_basename}"
        field_slices = piece_fields(z0, z1)
        write_piece_vti(
            field_slices, (z0, z1), whole_dims, spacing_tuple, origin_tuple, piece_path, compression
        )
        # Dropped before the next piece is built, so at most one piece's arrays
        # are ever resident — the whole point of the streaming writer.
        del field_slices
        total_size_bytes += os.path.getsize(piece_path)
        piece_specs.append((piece_rel, (z0, z1)))

    field_specs = [(name, _numpy_to_vtk_type_str(SAVE_DTYPE)) for name in field_names]
    write_pvti_master(
        output_path_pvti, whole_dims, spacing_tuple, origin_tuple, piece_specs, field_specs
    )

    return {
        "path_pvti": output_path_pvti,
        "pieces_dir": pieces_dir,
        "n_pieces": len(extents),
        "piece_extents_z": extents,
        "dimensions_xyz": [int(nx), int(ny), int(nz)],
        "spacing_um_xyz": list(spacing_tuple),
        "origin_um_xyz": list(origin_tuple),
        "fields": list(field_names),
        "dtype": np.dtype(SAVE_DTYPE).name,
        "total_size_MB": round(total_size_bytes / (1024**2), 2),
        "compression": "zlib" if compression else "none",
        "replace_nan": replace_nan,
        "write_valid_mask": write_valid_mask,
        "nan_sentinel": sentinel,
        "padded_fraction": nan_fraction_overall,
    }


def save_volumes_as_pvti(
    volumes: dict,
    spacing: tuple,
    output_path_pvti: str,
    *,
    origin: tuple = (0.0, 0.0, 0.0),
    n_pieces: int = 16,
    compression: bool = False,
    replace_nan: bool = True,
    write_valid_mask: bool = True,
    nan_sentinel: float | None = None,
) -> dict:
    """Write one or more co-registered (Z, Y, X) volumes to a partitioned VTI dataset."""
    if not volumes:
        raise ValueError("No volumes to save")
    if not output_path_pvti.endswith(".pvti"):
        raise ValueError(f"output_path_pvti must end with .pvti: {output_path_pvti}")

    shapes = {name: v.shape for name, v in volumes.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"All volumes must share the same shape, got {shapes}")
    nz, ny, nx = shapes[next(iter(shapes))]

    valid_mask = None
    sentinel = None
    nan_fraction_overall = 0.0
    if replace_nan or write_valid_mask:
        valid_mask = np.ones((nz, ny, nx), dtype=bool)
        for v in volumes.values():
            valid_mask &= np.isfinite(v)
        nan_fraction_overall = float(np.mean(~valid_mask))

    if replace_nan:
        if nan_sentinel is not None:
            sentinel = float(nan_sentinel)
        else:
            global_min, global_max = np.inf, -np.inf
            for v in volumes.values():
                f = v[np.isfinite(v)]
                if len(f):
                    global_min = min(global_min, float(f.min()))
                    global_max = max(global_max, float(f.max()))
            sentinel = _sentinel_for(global_min, global_max)

    cleaned_volumes = {}
    for name, vol in volumes.items():
        cleaned_volumes[name] = np.where(np.isfinite(vol), vol, sentinel) if replace_nan else vol
    if write_valid_mask:
        cleaned_volumes["valid_mask"] = valid_mask.astype(SAVE_DTYPE)

    return _write_partitioned_vti(
        lambda z0, z1: {name: vol[z0 : z1 + 1] for name, vol in cleaned_volumes.items()},
        list(cleaned_volumes.keys()),
        (nz, ny, nx),
        spacing,
        output_path_pvti,
        origin=origin,
        n_pieces=n_pieces,
        compression=compression,
        replace_nan=replace_nan,
        write_valid_mask=write_valid_mask,
        sentinel=sentinel,
        nan_fraction_overall=nan_fraction_overall,
    )


# -----------------------------------------------------------------------------
# Streaming PVTI writer
# -----------------------------------------------------------------------------
class _FieldStream:
    """One field's blocks, buffered so ascending, overlapping Z ranges can be served.

    :func:`compute_piece_extents_z` makes adjacent pieces share one Z index, so
    requests overlap and a consume-and-discard reader would drop a row it still
    needs. This keeps the buffered blocks that a pending request can still
    reach, and no more.
    """

    def __init__(self, provider) -> None:
        self._it = provider.blocks()
        self._buffer: list = []
        self._exhausted = False
        self._tail = tuple(int(d) for d in provider.shape[1:])
        # `StreamedAlignment.dtype` *declares* the block dtype, so it is the
        # right answer before a block exists — but the empty placeholder must
        # match the blocks a caller would otherwise get, so once one has been
        # seen the real dtype replaces the declared one rather than trusting
        # the two to agree.
        self._dtype = np.dtype(provider.dtype)

    def _prune(self, z0: int) -> None:
        self._buffer = [(sl, b) for sl, b in self._buffer if sl.stop > z0]

    def slab(self, z0: int, z1: int) -> np.ndarray:
        """This field over the half-open range ``[z0, z1)``, concatenated."""
        self._prune(z0)
        while not self._exhausted and (not self._buffer or self._buffer[-1][0].stop < z1):
            item = next(self._it, None)
            if item is None:
                self._exhausted = True
                break
            self._buffer.append(item)
        self._prune(z0)  # again: the fill may have pulled blocks that end below z0
        if self._buffer:
            self._dtype = self._buffer[0][1].dtype
        parts = [
            block[max(sl.start, z0) - sl.start : min(sl.stop, z1) - sl.start]
            for sl, block in self._buffer
            if min(sl.stop, z1) > max(sl.start, z0)
        ]
        # An empty Z axis (or a degenerate extent) covers no block at all; the
        # in-core writer slices an empty piece there rather than failing, so
        # match it instead of raising.
        if not parts:
            return np.empty((0,) + self._tail, dtype=self._dtype)
        # One part is a *slice of the buffered block*, so returning it directly
        # costs nothing where concatenating would copy. That matters most where
        # the copy is largest: a budget generous enough to leave one block per
        # field makes every slab a slice of a whole aligned volume, and copying
        # it holds a second one. Measured on a 128x192x192 four-field export at
        # an unbounded budget: 524 MiB peak RSS concatenating unconditionally,
        # 413 MiB with this.
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


class _SlabReader:
    """Serve ascending, possibly overlapping Z ranges of every field at once.

    One :class:`_FieldStream` per field rather than one interleaved walk,
    because **the providers do not block alike**: a centred CoM field carries
    the centring statistic's working set and a plain FWHM field does not, so
    :func:`~dfxm.common.alignment.align_volume_streamed` solves a different
    block size for each out of the same budget. Every field is on the same Z
    axis, which is what makes a shared ``[z0, z1)`` request meaningful; how each
    gets there is its own business.
    """

    def __init__(self, providers) -> None:
        self._streams = {name: _FieldStream(prov) for name, prov in providers.items()}

    @property
    def names(self) -> list:
        return list(self._streams)

    def field_slab(self, name: str, z0: int, z1: int) -> np.ndarray:
        """One field over the half-open range ``[z0, z1)``.

        Per field rather than all at once, so a caller that converts each slab
        and drops it never holds more than one of them.
        """
        return self._streams[name].slab(z0, z1)

    def slab(self, z0: int, z1: int) -> dict:
        """Every field over the half-open range ``[z0, z1)``."""
        return {name: stream.slab(z0, z1) for name, stream in self._streams.items()}


def _combined_finite_mask(fields: dict) -> np.ndarray:
    """Voxels finite in EVERY field — the in-core ``valid_mask &= isfinite(v)``."""
    mask = None
    for block in fields.values():
        finite = np.isfinite(block)
        mask = finite if mask is None else (mask & finite)
    return mask


def _z_grid_key(provider) -> bytes:
    """A hashable identity for a provider's Z grid, for the equality check."""
    return np.ascontiguousarray(provider.z_uniform_um, dtype=np.float64).tobytes()


# Bytes resident per voxel per field while a piece is written: the SAVE_DTYPE
# cleaned array, and VTK's deep copy of it inside `write_piece_vti`. Cleaning
# straight to SAVE_DTYPE (see `_clean_to_save_dtype`) and dropping each field's
# float64 slab as it converts is what got this down from 24 — a float64 slab, a
# float64 cleaned copy, a float32 contiguous copy and the VTK copy, all live at
# once for every field.
PIECE_BYTES_PER_VOXEL_PER_FIELD = 2 * np.dtype(SAVE_DTYPE).itemsize


def advisory_n_pieces(shape, n_fields: int, budget_bytes: int) -> int:
    """Fewest Z pieces whose per-piece residency fits *budget_bytes*.

    A ``.vti`` piece is written in one call, so ``n_pieces`` — not
    ``budget_bytes`` — is what bounds the piece pass, and too few pieces can
    make a *streamed* export cost more than the in-core one it replaced while
    every alignment block stays obediently inside its budget. Measured on a
    128x192x192 four-field export at a fixed 64 MB budget, against an in-core
    peak of 586-593 MB: ``n_pieces`` of 1 / 2 / 16 / 64 peaked at 641 / 476 /
    271 / 247 MB before the piece pass stopped holding every field's float64
    slab and float64 cleaned copy at once, and 559 / 408 / 261 / 244 MB after.
    The shape of the curve is what this function exists for — the residency is
    ``n_fields`` pieces however cheap each one is made.

    Advisory, never enforced: ``num_pieces_z`` is a user's choice about their
    pvserver's rank count and changing it silently would change the product.
    The caller warns instead.
    """
    nz, ny, nx = (int(d) for d in shape)
    per_piece_layer = max(1, ny * nx * max(1, int(n_fields)) * PIECE_BYTES_PER_VOXEL_PER_FIELD)
    layers_affordable = max(1, int(max(1, int(budget_bytes)) // per_piece_layer))
    # Ceiling division: enough pieces that a piece's layers fit the budget.
    return max(1, min(nz, -(-nz // layers_affordable)))


def _clean_to_save_dtype(block: np.ndarray, finite, sentinel) -> np.ndarray:
    """``np.where(finite, block, sentinel)`` in ``SAVE_DTYPE``, without the float64 copy.

    Byte-for-byte what the in-core writer produces: it builds the cleaned array
    in the block's own (float64) dtype and lets :func:`write_piece_vti` cast it,
    which rounds each finite value to ``SAVE_DTYPE`` and the sentinel with it.
    Rounding first and substituting after reaches the same array — and reaches
    it at half the width, on a piece where every field's copy is resident at
    once. ``finite=None`` means "no substitution" (``replace_nan=False``), where
    this is just the cast :func:`write_piece_vti` would have done anyway.

    Casting here rather than there also makes that function's
    ``np.ascontiguousarray(..., dtype=SAVE_DTYPE)`` a no-op instead of a second
    full-size allocation.
    """
    out = np.empty(block.shape, dtype=SAVE_DTYPE)
    np.copyto(out, block, casting="unsafe")
    if finite is not None:
        np.copyto(out, np.asarray(sentinel, dtype=SAVE_DTYPE), where=~finite)
    return out


def _survey(providers, *, count_invalid: bool, find_range: bool) -> tuple[int, float, float]:
    """One traversal for both global quantities a piece cannot compute itself.

    Returns ``(invalid_voxels, global_min, global_max)`` — the count of voxels
    non-finite in ANY field (the in-core ``valid_mask`` semantics) and the
    finite range across ALL fields. Deliberately one walk rather than two:
    ``StreamedAlignment.blocks`` is a factory and every call re-runs the whole
    alignment chain, so surveying the count and the range separately would
    align each field twice before the piece pass aligns it a third time. This
    is a time saving, not a memory one — the peak is set by how much is
    resident within a walk, not by how many walks there are.

    **Not skipped on the in-core rung**, and the judgement is deliberate. A
    whole-array path *can* reach both quantities directly — that is exactly what
    :func:`save_volumes_as_pvti` does — but doing so would give the sentinel and
    the padded fraction a *second* definition, chosen by how much memory the
    machine has. Both quantities reach the product (the sentinel is written into
    every padded voxel), so a divergence between the two definitions would be a
    machine-dependent difference in an exported file: precisely the failure
    `visualize` shipped twice (``~isnan`` vs ``isfinite``, ``np.nanmean`` vs
    ``stream_mean``) and had to repair at the streaming definition. What the
    in-core rung removes instead is the *re-alignment*: :func:`_drained` has
    already materialised each field, so this walk is one numpy pass over resident
    memory rather than a second traversal of the alignment chain. Measured on a
    128x192x192 four-field export, this pass costs **1.00 s of a 3.62 s streamed
    run (27.7%)** against **0.089 s of a 3.45 s in-core one (2.6%)**; an
    independent measurement on another machine put the in-core share at 0.158 s
    (4.5%) and the streamed one at 27.3%. So skipping it in-core buys at most
    ~4.5%, in exchange for a second definition of two numbers that reach the
    file. It is not worth it.
    """
    from ..common import volumeio

    nz = int(next(iter(providers.values())).shape[0])
    # Walk in slabs no larger than the smallest block any provider will yield,
    # so this pass never holds more than the piece pass does.
    step = max(1, min(int(prov.block_layers) for prov in providers.values()))
    reader = _SlabReader(providers)
    invalid = 0
    global_min, global_max = np.inf, -np.inf
    for z0 in range(0, nz, step):
        fields = reader.slab(z0, min(z0 + step, nz))
        if count_invalid:
            invalid += int(np.count_nonzero(~_combined_finite_mask(fields)))
        if find_range:
            for block in fields.values():
                # Per block, so the finite-value filtering is `volumeio`'s
                # definition rather than a second one here; min/max merge
                # across blocks exactly as `stream_minmax` merges them
                # internally.
                lo, hi = volumeio.stream_minmax([block])
                if np.isfinite(lo):
                    global_min = min(global_min, lo)
                    global_max = max(global_max, hi)
    return invalid, global_min, global_max


def save_volumes_streamed(
    providers: dict,
    spacing: tuple,
    output_path_pvti: str,
    *,
    origin: tuple = (0.0, 0.0, 0.0),
    n_pieces: int = 16,
    compression: bool = False,
    replace_nan: bool = True,
    write_valid_mask: bool = True,
    nan_sentinel: float | None = None,
) -> dict:
    """Write a partitioned VTI dataset from streamed fields, one piece at a time.

    ``providers`` maps field name to a
    :class:`~dfxm.common.alignment.StreamedAlignment`. Every provider must share
    one shape **and one Z grid** — they are co-registered by construction, so a
    mismatch is a bug rather than a case to handle, and both are checked because
    pairing the fields by absolute output Z index is the invariant that makes a
    per-field slab reader equivalent to one interleaved walk.

    **This function is on both rungs of the stage's ladder, and does not know
    which.** A provider whose blocks come from a resident array (:func:`_drained`,
    the in-core rung) is the same thing here as one whose blocks come from the
    alignment chain, so the two rungs cannot write different bytes — there is one
    writer, not two kept in step by hand. What the rung changes is the cost of
    *traversing* a provider: aligning a field again, or walking memory.

    Two passes. The first computes only what a piece cannot know locally: the
    NaN sentinel's global range and the overall invalid fraction. The second
    builds each piece's valid mask and cleaned arrays from that piece's Z-slab
    alone, so peak memory is one piece per field rather than one volume per
    field.

    **What this does and does not bound.** The aligned volume is never
    materialised, but a ``.vti`` piece is written in one call, so a piece's
    fields are resident together. The peak therefore scales as
    ``n_fields * volume_bytes / n_pieces`` — set by ``n_pieces`` (the stage's
    "Z pieces"), **not** by the providers' ``budget_bytes``, which sizes only
    the alignment blocks feeding the slab. More pieces means a lower peak; too
    few and the run can peak *above* what an in-core export would have cost,
    with the budget nominally honoured the whole time. :func:`advisory_n_pieces`
    is the floor that keeps that from happening silently.

    Returns the same dict as :func:`save_volumes_as_pvti`, and — for the same
    fields — writes byte-identical pieces.
    """
    if not providers:
        raise ValueError("No volumes to save")
    if not output_path_pvti.endswith(".pvti"):
        raise ValueError(f"output_path_pvti must end with .pvti: {output_path_pvti}")

    shapes = {name: tuple(prov.shape) for name, prov in providers.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"All volumes must share the same shape, got {shapes}")
    nz, ny, nx = shapes[next(iter(shapes))]
    grids = {name: (float(prov.scale_z_um), _z_grid_key(prov)) for name, prov in providers.items()}
    if len(set(grids.values())) != 1:
        raise ValueError(
            "All volumes must share one Z grid (scale_z_um and z_uniform_um); "
            f"got { ({n: g[0] for n, g in grids.items()}) } with "
            f"{len(set(g[1] for g in grids.values()))} distinct grids"
        )

    # --- pass 1: the two things a piece cannot compute for itself ---
    sentinel = None
    nan_fraction_overall = 0.0
    count_invalid = bool(replace_nan or write_valid_mask)
    find_range = bool(replace_nan and nan_sentinel is None)
    if count_invalid or find_range:
        invalid, global_min, global_max = _survey(
            providers, count_invalid=count_invalid, find_range=find_range
        )
        total = nz * ny * nx
        if count_invalid:
            nan_fraction_overall = (invalid / total) if total else 0.0
        if find_range:
            sentinel = _sentinel_for(global_min, global_max)
    if replace_nan and nan_sentinel is not None:
        sentinel = float(nan_sentinel)

    # --- pass 2: one piece at a time ---
    reader = _SlabReader(providers)

    def piece_fields(z0: int, z1: int) -> dict:
        # One field at a time, converting and dropping each float64 slab before
        # asking for the next: holding all of them to build the mask, and then a
        # float64 cleaned copy of each on top, is what made the peak scale with
        # `n_pieces` badly enough to beat the in-core writer at n_pieces=1.
        cleaned: dict = {}
        mask = None
        for name in reader.names:
            block = reader.field_slab(name, z0, z1 + 1)  # piece extents are inclusive
            finite = np.isfinite(block) if (write_valid_mask or replace_nan) else None
            if write_valid_mask:
                mask = finite if mask is None else (mask & finite)
            # Cleaning is per-field `isfinite`, the mask is the combined one.
            # That asymmetry is the in-core writer's and is preserved.
            cleaned[name] = _clean_to_save_dtype(block, finite if replace_nan else None, sentinel)
            del block, finite
        if write_valid_mask:
            cleaned["valid_mask"] = mask.astype(SAVE_DTYPE)
        return cleaned

    field_names = list(providers) + (["valid_mask"] if write_valid_mask else [])
    return _write_partitioned_vti(
        piece_fields,
        field_names,
        (nz, ny, nx),
        spacing,
        output_path_pvti,
        origin=origin,
        n_pieces=n_pieces,
        compression=compression,
        replace_nan=replace_nan,
        write_valid_mask=write_valid_mask,
        sentinel=sentinel,
        nan_fraction_overall=nan_fraction_overall,
    )


def estimate(params: dict) -> CostEstimate:
    """Peak memory for this run, from HDF5 shapes only.

    ``run()`` processes the mosaicity and strain volume files **sequentially**
    via separate helper calls (``_process_mosaicity`` then ``_process_strain``)
    — each helper's locals die when it returns, so the two files' peaks do not
    add, and the run's peak is the max over the (at most two) files processed.
    ``chunkable=True``.

    **The arithmetic below over-predicts both rungs, and deliberately so.** It
    prices the export as it stood before this phase: every raw field resident at
    once, an aligned float64 copy of every field accumulated, and a second
    ``np.where``-cleaned set plus a boolean mask built whole before the first
    piece is written. Neither rung does the last of those any more — the piece
    writer builds each `.vti` from its own Z-slab on both — and the streaming
    rung does not hold the aligned copies either. It is what
    ``advice.plan_run`` compares against the machine's headroom, and an
    over-estimate there only makes it hand over a *smaller* ``budget_bytes`` —
    which is also what decides the rung, so over-predicting biases the stage
    toward streaming: slower, never an OOM. An under-estimate would let a run
    take the in-core rung and then OOM. Recalibrating it means measuring both
    rungs' peaks on the real dataset, not editing the terms below by inspection —
    the warning at the end of this docstring applies unchanged.

    The terms: the aligned float64 copy of every field (``apply_roi_3d`` ->
    ``apply_samy_shifts_to_volume`` -> ``interpolate_to_uniform_z``, the last of
    which upcasts to float64) accumulated for the whole export —
    ``file_elems * 8`` — and a further ``np.where``-cleaned float64 copy of
    every field plus a boolean ``valid_mask`` before downcasting to
    ``SAVE_DTYPE`` per piece at write time, bounded as ``file_elems * 8 +
    largest_elems * 8``.

    **Recalibration warning — do not just delete the ``file_total`` term.** The
    current figure over-predicts on the mosaicity file, which is the safe
    direction, and is deliberately left unchanged here. The mosaicity saving is
    only ``file_total`` minus one field's raw bytes, and the two
    ``file_elems * 8`` terms that remain are expressed in *unpadded* elements:
    ``apply_samy_shifts_to_volume`` widens the canvas along image-X by the
    extreme samy offsets and ``interpolate_to_uniform_z`` resamples onto a grid
    that exceeds the layer count whenever samz is irregular, and the two
    inflations multiply — so every aligned copy is *larger* than
    ``file_elems``. Neither extent is derivable from HDF5 shapes alone (both
    depend on motor values), so the retired term is in part accidental headroom
    covering that; removing it without replacing it can turn an over-estimate
    into an under-estimate, which is the dangerous direction (it greenlights a
    run that then OOMs).
    """
    p = {**STAGE.defaults(), **params}
    total = 0
    peak = 0
    largest: tuple[int, ...] | None = None
    for name in ("mosa_volume_file", "strain_volume_file"):
        path = str(p.get(name) or "")
        if not path:
            continue
        file_total, shape, itemsize = sum_dataset_bytes(path)
        if not file_total:
            continue
        total += file_total
        file_elems = file_total // max(1, itemsize)
        largest_elems = 1
        if shape is not None:
            for dim in shape:
                largest_elems *= dim
            if largest is None or len(shape) > len(largest):
                largest = shape
        file_peak = file_total + 2 * file_elems * 8 + largest_elems * 8
        peak = max(peak, file_peak)
    if not total:
        return CostEstimate(0, 0, None, True, "no readable volume files selected yet")
    return CostEstimate(peak, total, largest, True, None)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def mosa_field_names(filepath: str) -> list[str]:
    """Field names in a mosaicity volume file, without reading any data.

    Sorted, so the field order is deterministic instead of inheriting h5py's
    group-key order.
    """
    names = []
    with h5py.File(filepath, "r") as f:
        for group in ("chi", "mu"):
            if group in f:
                for ds in f[group].keys():
                    names.append(f"{group}_{ds.replace(' ', '_')}")
    return sorted(names)


def mosa_dataset(f, name: str):
    """The open HDF5 dataset for *name* in an already-open file, or None.

    The name-matching convention lives here alone: a streaming caller needs the
    dataset (to slice it block by block) rather than its contents, and
    :func:`load_mosa_field` is the read-it-all-now wrapper over the same lookup.
    """
    for group in ("chi", "mu"):
        if group not in f:
            continue
        for ds in f[group].keys():
            if f"{group}_{ds.replace(' ', '_')}" == name:
                return f[group][ds]
    return None


def load_mosa_field(filepath: str, name: str):
    """One field from a mosaicity volume file, or None if absent."""
    with h5py.File(filepath, "r") as f:
        dset = mosa_dataset(f, name)
        return None if dset is None else dset[:]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _parse_pair(text) -> tuple | None:
    if text is None or str(text).strip() == "":
        return None
    parts = [int(v) for v in str(text).replace(" ", "").split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'a,b', got {text!r}")
    return tuple(parts)


def _motors(raw_root, pattern, samy_path, samz_path):
    if not raw_root or not pattern:
        return np.array([]), np.array([])
    folders = find_matching_folders(raw_root, pattern)
    if not folders:
        return np.array([]), np.array([])
    samy, samz, _ = extract_motor_positions(folders, samy_path, samz_path)
    return samy, samz


def _pvti_kwargs(p: dict) -> dict:
    return dict(
        n_pieces=int(p["num_pieces_z"]),
        compression=bool(p["piece_compression"]),
        replace_nan=bool(p["replace_nan"]),
        write_valid_mask=bool(p["write_valid_mask"]),
    )


# Where a median centring caches the aligned volume when it cannot hold it in
# RAM. Its own subdirectory of the output folder, not the output folder itself:
# the file is multi-GB on a real dataset and would otherwise sit next to the
# user's PVTI looking like a product. `volumeio.scratch_array` removes it.
SCRATCH_SUBDIR = ".dfxm_scratch"


def _scratch_dir_for(out_dir: str, needed_bytes: int, notes: list) -> str | None:
    """A scratch directory under *out_dir*, or ``None`` when the disk cannot take it.

    ``None`` is not a failure: :func:`~dfxm.common.alignment.align_volume_streamed`
    falls back to re-running the alignment for each of the median's passes —
    slower, never refused, which is this project's governing rule. Only a
    multi-pass statistic ever caches, so a mean-centred (or uncentred) run never
    reaches here and never creates the directory.
    """
    from ..common import machine

    free = machine.probe_disk(out_dir)
    if free and needed_bytes and free < needed_bytes:
        notes.append(
            f"median centring will re-read instead of caching: {SCRATCH_SUBDIR} would need "
            f"about {needed_bytes // (1 << 20)} MB but only {free // (1 << 20)} MB is free "
            f"on {out_dir} — slower, same result"
        )
        return None
    path = os.path.join(out_dir, SCRATCH_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _multipass_scratch(center_method, out_dir: str, dset, notes: list) -> str | None:
    """Scratch directory for a centring that traverses more than once, else ``None``.

    Only ``median`` caches; ``mean`` is a single pass and ``None`` is both
    correct and free. One field caches at a time (the cache is released before
    the next provider is built), so the ask is one aligned float64 volume,
    floored at the input's own element count — the aligned copy is *larger* (the
    samy X-pad and the uniform-Z resample both inflate), so this under-states
    the need and the check is a coarse guard against an obviously full disk
    rather than a guarantee.
    """
    if not center_method or str(center_method).lower() != "median":
        return None
    elems = 1
    for dim in dset.shape:
        elems *= int(dim)
    return _scratch_dir_for(out_dir, elems * 8, notes)


def _fits_in_core(providers: dict) -> bool:
    """Whether every field's share of the budget holds its whole aligned volume.

    The same question :func:`~dfxm.common.advice.plan_run` asks, in the budget's
    own working-set currency: :func:`~dfxm.common.alignment.align_volume_streamed`
    already solved how many output layers one block may carry, so "does it fit?"
    is "is that the whole Z axis?". Nothing is read to answer it — the blocking
    is known before a voxel is touched.

    **All or nothing across the set.** A ``.vti`` piece carries every field, so
    the writer holds one stream per field and their working sets coexist; the
    budget is divided by that count at the call site and a single field short of
    its share sends the whole export streaming. The fields genuinely do not block
    alike — a centred CoM field carries the centring statistic's working set and
    a plain FWHM field does not — so this is a case that occurs rather than a
    defensive nicety.

    No providers is not "it fits": there is nothing to materialise and nothing to
    write.
    """
    return bool(providers) and all(
        int(prov.block_layers) >= int(prov.shape[0]) for prov in providers.values()
    )


def _drained(provider):
    """*provider* with its aligned volume materialised, as a one-block provider.

    **The in-core rung, and the only function on it** — which is what lets a test
    pin which rung ran instead of merely observing that the run finished.

    Streaming is a fallback for insufficient memory, not a product improvement.
    Every export used to pay for it: the aligned volume was never materialised,
    so :func:`_survey` and the piece pass each re-ran the whole alignment chain
    (``blocks`` is a factory) and the export aligned every field **twice**,
    measured at 1.2-1.7x the wall clock of the in-core exporter this replaced.
    Draining once collapses that back to one alignment; both passes then walk
    resident memory. Measured over three sizes of a four-field mosaicity export,
    the old unconditional stream at the machine's own budget against this rung:
    0.117 -> 0.095 s (1 MB/volume), 0.961 -> 0.727 s (8 MB), 5.045 -> 3.447 s
    (36 MB) — 1.23x to 1.46x, and 1.42x in an independent measurement.

    **The recovery is not uniform across budgets, and the old range should not
    be repeated as if it were.** The unconditional stream was slow specifically
    at a *large* budget, where each field is one huge block aligned twice with
    nothing staying in cache: on the 36 MB export it cost 5.045 s at the
    machine's budget but only 3.625 s at 1 GiB and 3.549 s at 64 MiB. So the
    workstation — the machine that needed no streaming at all — was paying the
    most, which is what makes this the right rung to add; a memory-constrained
    machine was already near the in-core time.

    The writer is unchanged and unaware. It consumes providers, and a provider
    whose blocks come from an array is the same thing to it as one whose blocks
    come from the alignment chain — which is why the two rungs cannot produce
    different bytes. That is the shape `slices` used (one implementation per
    statistic, the rung decides only where blocks come from) rather than the one
    `visualize` had to repair twice (in-core and streaming siblings kept in step
    by hand).
    """
    data = A.materialise_blocks(provider.blocks, provider.shape, provider.dtype)
    whole = slice(0, int(provider.shape[0]))
    return replace(
        provider,
        # The declared dtype is what the blocks were promised to be; the array is
        # what they turned out to be. `_FieldStream` makes the same correction
        # for the same reason.
        dtype=np.dtype(data.dtype),
        block_layers=int(provider.shape[0]),
        working_set_bytes=int(data.nbytes),
        blocks=lambda: iter([(whole, data)]),
    )


def _writable_providers(
    providers: dict, *, budget_bytes: int, n_pieces: int, write_valid_mask: bool, label: str, notes
):
    """Pick the rung and return the providers :func:`save_volumes_streamed` writes from.

    In-core when the budget holds every field's whole aligned volume, streaming
    when it does not — :func:`~dfxm.common.advice.plan_run`'s own choice, made
    per export.

    **The Z-piece advisory is raised on both rungs**, because the piece pass is
    the same on both and is bounded by ``n_pieces`` rather than by the budget on
    both. It was briefly suppressed in-core on the argument that a piece must be
    a fraction of a volume set that is resident anyway — which is true and beside
    the point: the piece's fields are held *on top of* whatever the alignment
    left resident. Measured on a 128x192x192 four-field export, 16 pieces against
    one: **413-421 -> 566 MiB in-core** (a 2 GiB budget) and **493 -> 633 MiB
    streamed** (1 GiB). The piece pass adds ~145 MiB at ``n_pieces = 1`` on
    *either* rung, which is what makes gating the advisory on the rung wrong.

    On a real in-core run the advisory nonetheless never fires, and that is
    arithmetic rather than suppression: taking the rung requires each field's
    share of the budget to hold a whole alignment working set (~430 MB here
    against a 36 MB volume, so ~5-12x), which already leaves room for one piece
    of every field. The tests therefore exercise the ungated path by calling this
    function directly.
    """
    shape = next(iter(providers.values())).shape if providers else None
    if shape is not None:
        # `valid_mask` is written as a field of its own, so it is resident in the
        # piece alongside the data fields and is counted like one.
        n_fields = len(providers) + (1 if write_valid_mask else 0)
        note = _piece_advice(shape, n_fields, budget_bytes, n_pieces, label)
        if note:
            notes.append(note)
    if _fits_in_core(providers):
        return {name: _drained(prov) for name, prov in providers.items()}
    return providers


def _piece_advice(shape, n_fields: int, budget_bytes: int, n_pieces: int, label: str) -> str | None:
    """The warning for a Z-piece count too low to keep the piece pass bounded."""
    advised = advisory_n_pieces(shape, n_fields, budget_bytes)
    if n_pieces >= advised:
        return None
    per_piece = (
        int(shape[1]) * int(shape[2]) * max(1, n_fields) * PIECE_BYTES_PER_VOXEL_PER_FIELD
    ) * -(-int(shape[0]) // max(1, n_pieces))
    return (
        f"{label}: Z pieces = {n_pieces} needs about {per_piece // (1 << 20)} MB for one "
        f"piece of every field, over this machine's {budget_bytes // (1 << 20)} MB of "
        f"headroom — raise Z pieces to {advised} or more. The memory budget bounds the "
        "alignment, not the piece: one piece of every field is held on top of it, whether "
        "the volumes were aligned in one go or streamed."
    )


# The Z step (µm) the export falls back to when no raw scan folders were found,
# so there are no samz positions to derive one from. `extract_motor_positions`
# fills samy and samz from the same folders, so they are empty together and this
# case means "no motors at all", not "no Z motor".
_NO_MOTOR_Z_STEP_UM = 2.0


def _whole_volume_stream(data: np.ndarray, center_offset: float = 0.0):
    """An in-memory array presented as a one-block ``StreamedAlignment``.

    So the writer has exactly one kind of input to consume. Used for the
    no-motor case below, where there is nothing to stream in the first place.
    """
    return A.StreamedAlignment(
        shape=tuple(int(d) for d in data.shape),
        dtype=np.dtype(data.dtype),
        z_uniform_um=np.arange(data.shape[0], dtype=float) * _NO_MOTOR_Z_STEP_UM,
        scale_z_um=_NO_MOTOR_Z_STEP_UM,
        pad_left=0,
        pad_right=0,
        center_offset=float(center_offset),
        block_layers=int(data.shape[0]),
        working_set_bytes=int(data.nbytes),
        blocks=lambda: iter([(slice(0, int(data.shape[0])), data)]),
    )


def _unaligned_field(dset, *, roi_x, roi_y, take_abs: bool, center_method: str | None):
    """The no-motor export path, as a provider: no shift, no Z resampling.

    ``_motors`` returns empty ``samy`` **and** ``samz`` together when the
    raw-folder glob matched nothing, and the stage has always answered that by
    skipping both motor-driven steps and labelling the layers
    ``_NO_MOTOR_Z_STEP_UM`` apart. That cannot be expressed through
    :func:`~dfxm.common.alignment.align_volume_streamed`, which always
    interpolates: resampling a NaN-bearing volume onto its own Z nodes is not
    the identity, because scipy's linear interpolant reads the value *below*
    each node and so spreads every NaN one layer down — measured here as 1299
    of 9360 voxels changing their ``valid_mask``. So this keeps the old chain
    (abs -> ROI -> centre) and hands the result over as a single block.

    Nothing is streamed, which costs what it always did: one volume in memory.
    The saving that still applies is the writer's — the cleaned copy and the
    mask are built per piece either way. A run with no motor positions exports
    an unaligned volume, so it is a misconfigured run rather than the large
    production run this phase is about.
    """
    raw = dset[:]
    data = A.apply_roi_3d(np.abs(raw) if take_abs else raw, roi_x, roi_y)
    del raw
    offset = 0.0
    if center_method:
        data, offset = A.center_around_zero(data, center_method)
    return _whole_volume_stream(data, offset)


# -----------------------------------------------------------------------------
# Per-volume processing
# -----------------------------------------------------------------------------
def _process_mosaicity(
    p, out_dir, scale_x, scale_y, samy_dir, roi_x, roi_y, *, budget_bytes, notes
) -> ExportInfo | None:
    names = mosa_field_names(p["mosa_volume_file"])
    if not names:
        return None
    samy, samz = _motors(p["raw_root"], p["mosa_pattern"], p["samy_path"], p["samz_path"])

    with h5py.File(p["mosa_volume_file"], "r") as f:
        providers = {}
        # Every field's stream is live at once — a `.vti` piece carries all of
        # them, so `save_volumes_streamed` holds one open per field and their
        # working sets coexist. Hence a share of the budget each. This divides
        # by the number of CONCURRENT streams, which is a fact about this call
        # site; it is not a correction to what `budget_bytes` means.
        per_field = max(1, int(budget_bytes) // max(1, len(names)))
        for name in names:
            dset = mosa_dataset(f, name)
            if dset is None:
                continue
            is_com = "Center_of_mass" in name
            is_fwhm = "FWHM" in name
            take_abs = is_fwhm and bool(p["abs_mosa_fwhm"])
            center_method = p["center_method"] if is_com and bool(p["center_mosa_com"]) else None
            providers[name] = (
                A.align_volume_streamed(
                    dset,
                    samy,
                    samz,
                    scale_x=scale_x,
                    samy_direction=samy_dir,
                    roi_x=roi_x,
                    roi_y=roi_y,
                    take_abs=take_abs,
                    center_method=center_method,
                    budget_bytes=per_field,
                    scratch_dir=_multipass_scratch(center_method, out_dir, dset, notes),
                )
                if len(samz) > 0
                else _unaligned_field(
                    dset,
                    roi_x=roi_x,
                    roi_y=roi_y,
                    take_abs=take_abs,
                    center_method=center_method,
                )
            )

        first = next(iter(providers.values()), None)
        origin = (
            A.raw_detector_origin(
                samy,
                first.z_uniform_um if first is not None else None,
                scale_x=scale_x,
                scale_y=scale_y,
                roi_x=roi_x,
                roi_y=roi_y,
                darfix_origin_xy=_parse_pair(p["mosa_darfix_origin_xy"]) or (0, 0),
                samy_direction=samy_dir,
            )
            if bool(p["anchor_origin_to_reference"])
            else (0.0, 0.0, 0.0)
        )
        if len({tuple(prov.shape) for prov in providers.values()}) != 1:
            return None  # shape mismatch — skip merged export (rare)
        spacing = (scale_x, scale_y, first.scale_z_um)
        out_path = os.path.join(out_dir, "mosaicity_volume.pvti")
        pvti = _pvti_kwargs(p)
        providers = _writable_providers(
            providers,
            budget_bytes=budget_bytes,
            n_pieces=pvti["n_pieces"],
            write_valid_mask=pvti["write_valid_mask"],
            label="mosaicity",
            notes=notes,
        )
        info = save_volumes_streamed(providers, spacing, out_path, origin=origin, **pvti)
    return ExportInfo(
        "mosaicity",
        out_path,
        tuple(info["dimensions_xyz"]),
        tuple(info["spacing_um_xyz"]),
        tuple(info["origin_um_xyz"]),
        info["n_pieces"],
        info["fields"],
    )


def _process_strain(
    p, out_dir, scale_x, scale_y, samy_dir, roi_x, roi_y, *, budget_bytes, notes
) -> ExportInfo | None:
    samy, samz = _motors(p["raw_root"], p["strain_pattern"], p["samy_path"], p["samz_path"])
    with h5py.File(p["strain_volume_file"], "r") as f:
        dset = f["strain"] if "strain" in f else None
        if dset is None:
            return None
        center_method = p["center_method"] if bool(p["center_strain"]) else None
        provider = (
            A.align_volume_streamed(
                dset,
                samy,
                samz,
                scale_x=scale_x,
                samy_direction=samy_dir,
                roi_x=roi_x,
                roi_y=roi_y,
                center_method=center_method,
                budget_bytes=int(budget_bytes),
                scratch_dir=_multipass_scratch(center_method, out_dir, dset, notes),
            )
            if len(samz) > 0
            else _unaligned_field(
                dset, roi_x=roi_x, roi_y=roi_y, take_abs=False, center_method=center_method
            )
        )
        origin = (
            A.raw_detector_origin(
                samy,
                provider.z_uniform_um,
                scale_x=scale_x,
                scale_y=scale_y,
                roi_x=roi_x,
                roi_y=roi_y,
                darfix_origin_xy=_parse_pair(p["strain_darfix_origin_xy"]) or (0, 0),
                samy_direction=samy_dir,
            )
            if bool(p["anchor_origin_to_reference"])
            else (0.0, 0.0, 0.0)
        )
        spacing = (scale_x, scale_y, provider.scale_z_um)
        out_path = os.path.join(out_dir, "strain_volume.pvti")
        pvti = _pvti_kwargs(p)
        providers = _writable_providers(
            {"strain": provider},
            budget_bytes=budget_bytes,
            n_pieces=pvti["n_pieces"],
            write_valid_mask=pvti["write_valid_mask"],
            label="strain",
            notes=notes,
        )
        info = save_volumes_streamed(providers, spacing, out_path, origin=origin, **pvti)
    return ExportInfo(
        "strain",
        out_path,
        tuple(info["dimensions_xyz"]),
        tuple(info["spacing_um_xyz"]),
        tuple(info["origin_um_xyz"]),
        info["n_pieces"],
        info["fields"],
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def _run_budget_bytes(p: dict, out_dir: str) -> int:
    """Working-set budget for this run's alignment streams, in bytes.

    Measured from the machine unless the caller injected ``_budget_bytes``. The
    underscore marks it as not a :class:`StageSpec` parameter: it never appears
    on the form and is not part of the saved config, exactly like the
    ``plot_style`` snapshot ``gui/stage_view.py`` injects at run time. Tests use
    it to pin a blocking that does not depend on the machine they run on.

    The number is in ``tracemalloc``/allocation currency, which is what
    :func:`~dfxm.common.alignment.align_volume_streamed` prices its working set
    in — deliberately *not* RSS, which additionally carries the interpreter, VTK
    and h5py's buffers. So the machine's headroom, which *is* RSS, goes through
    :func:`~dfxm.common.advice.working_set_budget_bytes` with this stage's own
    :data:`RSS_FLOOR_BYTES` rather than straight in. An injected
    ``_budget_bytes`` is taken as already being in working-set currency, since a
    caller naming that key is naming the budget itself.
    """
    injected = p.get("_budget_bytes")
    if injected is not None:
        return max(1, int(injected))
    from ..common import advice, machine

    return advice.working_set_budget_bytes(
        machine.profile(output_dir=out_dir), rss_floor_bytes=RSS_FLOOR_BYTES
    )


def run(params: dict, progress: ProgressFn | None = None) -> ParaviewResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    if p["center_method"].lower() not in ("mean", "median"):
        raise ValueError(f"center_method must be mean/median (got {p['center_method']!r})")
    scale_x, scale_y = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    samy_dir = int(p["samy_direction"])
    roi_x, roi_y = _parse_pair(p["roi_x"]), _parse_pair(p["roi_y"])

    out_dir = p["output_dir"] or os.path.join(
        os.path.dirname(p["mosa_volume_file"] or p["strain_volume_file"] or "."), "paraview_exports"
    )
    os.makedirs(out_dir, exist_ok=True)
    budget_bytes = _run_budget_bytes(p, out_dir)
    result = ParaviewResult(output_dir=out_dir)

    if bool(p["export_mosaicity"]):
        mosa_file = p["mosa_volume_file"]
        if mosa_file and os.path.exists(mosa_file):
            progress(0.1, "exporting mosaicity volume")
            info = _process_mosaicity(
                p,
                out_dir,
                scale_x,
                scale_y,
                samy_dir,
                roi_x,
                roi_y,
                budget_bytes=budget_bytes,
                notes=result.notes,
            )
            if info:
                result.exports.append(info)
            else:
                result.skipped.append("mosaicity: no datasets / shape mismatch")
        elif mosa_file:
            result.skipped.append(f"mosaicity volume not found: {mosa_file}")

    if bool(p["export_strain"]):
        strain_file = p["strain_volume_file"]
        if strain_file and os.path.exists(strain_file):
            progress(0.55, "exporting strain volume")
            info = _process_strain(
                p,
                out_dir,
                scale_x,
                scale_y,
                samy_dir,
                roi_x,
                roi_y,
                budget_bytes=budget_bytes,
                notes=result.notes,
            )
            if info:
                result.exports.append(info)
            else:
                result.skipped.append("strain: 'strain' dataset not found")
        elif strain_file:
            result.skipped.append(f"strain volume not found: {strain_file}")

    info_path = os.path.join(out_dir, "export_info.txt")
    summary = {
        "exports": [
            dict(
                name=e.name,
                pvti=e.pvti_path,
                dimensions_xyz=e.dimensions_xyz,
                spacing_um_xyz=e.spacing_um_xyz,
                origin_um_xyz=e.origin_um_xyz,
                n_pieces=e.n_pieces,
                fields=e.fields,
            )
            for e in result.exports
        ],
        "skipped": result.skipped,
        "notes": result.notes,
        "config": {
            k: p[k] for k in ("center_method", "anchor_origin_to_reference", "num_pieces_z")
        },
    }
    with open(info_path, "w") as fh:
        fh.write(json.dumps(summary, indent=2, default=str) + "\n")
    # `volumeio.scratch_array` deletes its file; the directory it lived in is
    # ours. `rmdir` is exactly the semantics wanted — it removes an empty
    # directory and refuses a non-empty one, so a stray file (a crashed earlier
    # run, something a user put there) is never deleted.
    with contextlib.suppress(OSError):
        os.rmdir(os.path.join(out_dir, SCRATCH_SUBDIR))
    for note in result.notes:
        progress(0.98, note)
    result.info_path = info_path

    progress(1.0, f"exported {len(result.exports)} volume(s) -> {out_dir}")
    return result


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Export aligned volumes to ParaView PVTI.")
    ap.add_argument("--mosa-volume-file", default="")
    ap.add_argument("--strain-volume-file", default="")
    ap.add_argument("--raw-root", default="")
    ap.add_argument("--mosa-pattern", default="*")
    ap.add_argument("--strain-pattern", default="*")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--num-pieces-z", type=int, default=16)
    args = ap.parse_args(argv)
    res = run(
        dict(
            mosa_volume_file=args.mosa_volume_file,
            strain_volume_file=args.strain_volume_file,
            raw_root=args.raw_root,
            mosa_pattern=args.mosa_pattern,
            strain_pattern=args.strain_pattern,
            output_dir=args.output_dir,
            num_pieces_z=args.num_pieces_z,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(
        f"\nexported {len(res.exports)} volume(s) -> {res.output_dir}; skipped {len(res.skipped)}"
    )
    return 0


def roi_previews(params: dict) -> list:
    """(label, thunk) ROI-picker previews from the stacked mosa/strain volume(s)."""
    from ..common.figures import stacked_volume_previews

    return stacked_volume_previews(params)


if __name__ == "__main__":
    raise SystemExit(_main())
