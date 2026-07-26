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

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import numpy as np

from ..common import alignment as A
from ..common.raster import extract_motor_positions
from ..common.sort import find_matching_folders
from ..config.models import Param, ParamType, StageSpec

ProgressFn = Callable[[float, str], None]

SAVE_DTYPE = np.float32  # float32 is plenty for visualisation


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
            sentinel = (
                -1e30
                if not np.isfinite(global_min)
                else global_min - 1000.0 * max(global_max - global_min, 1.0)
            )

    cleaned_volumes = {}
    for name, vol in volumes.items():
        cleaned_volumes[name] = np.where(np.isfinite(vol), vol, sentinel) if replace_nan else vol
    if write_valid_mask:
        cleaned_volumes["valid_mask"] = valid_mask.astype(SAVE_DTYPE)

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
        field_slices = {name: vol[z0 : z1 + 1] for name, vol in cleaned_volumes.items()}
        write_piece_vti(
            field_slices, (z0, z1), whole_dims, spacing_tuple, origin_tuple, piece_path, compression
        )
        total_size_bytes += os.path.getsize(piece_path)
        piece_specs.append((piece_rel, (z0, z1)))

    field_specs = [(name, _numpy_to_vtk_type_str(SAVE_DTYPE)) for name in cleaned_volumes]
    write_pvti_master(
        output_path_pvti, whole_dims, spacing_tuple, origin_tuple, piece_specs, field_specs
    )

    info = {
        "path_pvti": output_path_pvti,
        "pieces_dir": pieces_dir,
        "n_pieces": len(extents),
        "piece_extents_z": extents,
        "dimensions_xyz": [int(nx), int(ny), int(nz)],
        "spacing_um_xyz": list(spacing_tuple),
        "origin_um_xyz": list(origin_tuple),
        "fields": list(cleaned_volumes.keys()),
        "dtype": np.dtype(SAVE_DTYPE).name,
        "total_size_MB": round(total_size_bytes / (1024**2), 2),
        "compression": "zlib" if compression else "none",
        "replace_nan": replace_nan,
        "write_valid_mask": write_valid_mask,
        "nan_sentinel": sentinel,
        "padded_fraction": nan_fraction_overall,
    }
    return info


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_mosa_datasets(filepath: str) -> dict:
    out = {}
    with h5py.File(filepath, "r") as f:
        for group in ("chi", "mu"):
            if group in f:
                for ds in f[group].keys():
                    out[f"{group}_{ds.replace(' ', '_')}"] = f[group][ds][:]
    return out


def load_strain_volume(filepath: str):
    with h5py.File(filepath, "r") as f:
        return f["strain"][:] if "strain" in f else None


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


# -----------------------------------------------------------------------------
# Per-volume processing
# -----------------------------------------------------------------------------
def _process_mosaicity(p, out_dir, scale_x, scale_y, samy_dir, roi_x, roi_y) -> ExportInfo | None:
    datasets = load_mosa_datasets(p["mosa_volume_file"])
    if not datasets:
        return None
    samy, samz = _motors(p["raw_root"], p["mosa_pattern"], p["samy_path"], p["samz_path"])

    processed = {}
    scale_z = None
    z_positions = None
    for name, raw in datasets.items():
        is_com = "Center_of_mass" in name
        is_fwhm = "FWHM" in name
        if is_fwhm and bool(p["abs_mosa_fwhm"]):
            raw = np.abs(raw)
        data = A.apply_roi_3d(raw, roi_x, roi_y)
        if len(samy) > 0:
            data = A.apply_samy_shifts_to_volume(data, samy, scale_x, samy_dir)
        if len(samz) > 0:
            data, z_pos, sz = A.interpolate_to_uniform_z(data, samz)
        else:
            sz = 2.0
            z_pos = np.arange(data.shape[0], dtype=float) * sz
        if scale_z is None:
            scale_z, z_positions = sz, z_pos
        if is_com and bool(p["center_mosa_com"]):
            data, _ = A.center_around_zero(data, p["center_method"])
        processed[name] = data

    origin = (
        A.raw_detector_origin(
            samy,
            z_positions,
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
    if len({v.shape for v in processed.values()}) != 1:
        return None  # shape mismatch — skip merged export (rare)
    spacing = (scale_x, scale_y, scale_z)
    out_path = os.path.join(out_dir, "mosaicity_volume.pvti")
    info = save_volumes_as_pvti(processed, spacing, out_path, origin=origin, **_pvti_kwargs(p))
    return ExportInfo(
        "mosaicity",
        out_path,
        tuple(info["dimensions_xyz"]),
        tuple(info["spacing_um_xyz"]),
        tuple(info["origin_um_xyz"]),
        info["n_pieces"],
        info["fields"],
    )


def _process_strain(p, out_dir, scale_x, scale_y, samy_dir, roi_x, roi_y) -> ExportInfo | None:
    strain = load_strain_volume(p["strain_volume_file"])
    if strain is None:
        return None
    samy, samz = _motors(p["raw_root"], p["strain_pattern"], p["samy_path"], p["samz_path"])
    data = A.apply_roi_3d(strain, roi_x, roi_y)
    if len(samy) > 0:
        data = A.apply_samy_shifts_to_volume(data, samy, scale_x, samy_dir)
    if len(samz) > 0:
        data, z_positions, scale_z = A.interpolate_to_uniform_z(data, samz)
    else:
        scale_z = 2.0
        z_positions = np.arange(data.shape[0], dtype=float) * scale_z
    if bool(p["center_strain"]):
        data, _ = A.center_around_zero(data, p["center_method"])
    origin = (
        A.raw_detector_origin(
            samy,
            z_positions,
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
    spacing = (scale_x, scale_y, scale_z)
    out_path = os.path.join(out_dir, "strain_volume.pvti")
    info = save_volumes_as_pvti(
        {"strain": data}, spacing, out_path, origin=origin, **_pvti_kwargs(p)
    )
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
    result = ParaviewResult(output_dir=out_dir)

    if bool(p["export_mosaicity"]):
        mosa_file = p["mosa_volume_file"]
        if mosa_file and os.path.exists(mosa_file):
            progress(0.1, "exporting mosaicity volume")
            info = _process_mosaicity(p, out_dir, scale_x, scale_y, samy_dir, roi_x, roi_y)
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
            info = _process_strain(p, out_dir, scale_x, scale_y, samy_dir, roi_x, roi_y)
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
        "config": {
            k: p[k] for k in ("center_method", "anchor_origin_to_reference", "num_pieces_z")
        },
    }
    with open(info_path, "w") as fh:
        fh.write(json.dumps(summary, indent=2, default=str) + "\n")
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
