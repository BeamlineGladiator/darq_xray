"""Concat stage — combine BLISS scans into a darfix-compatible entry.

A faithful port of the two legacy scripts:

* ``concatenate_h5_scans_v3.py`` (single folder)
* ``batch_concatenate_h5_scans_v1.py`` (walk a root, process each matching folder)

Both selected the ``*.1`` entries of a BLISS ``.h5`` and wrote a single
``entry_0000`` with a concatenated detector stack plus merged positioners:

    entry_0000/
        instrument/positioners/{mu, ccmth, ...}   # arrays / scalars
        measurement/pco_ff                        # detector stack (VDS or copy)

Positioner handling (preserved exactly):
  * array motors (e.g. ``mu``, N values per scan) are concatenated directly;
  * scalar motors that vary between scans (e.g. ``ccmth``) are expanded to one
    value per frame, then concatenated;
  * motors identical across every scan collapse back to a single scalar.

New vs the scripts: a ``copy_data`` toggle. Default ``False`` writes a Virtual
Dataset that references the original files (fast, fragile if originals move);
``True`` copies the detector frames into a self-contained output.

The stage exposes ``run(params, progress=None) -> ConcatResult`` and a headless
``__main__``; ``progress`` defaults to a no-op so it stays CLI/test friendly.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import numpy as np

from ..common import h5io
from ..common.sort import find_matching_folders
from ..config.models import Param, ParamType, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    """Default progress sink — does nothing (headless / tests)."""


# -----------------------------------------------------------------------------
# Parameter schema
# -----------------------------------------------------------------------------
STAGE = StageSpec(
    name="concat",
    label="Concatenate scans",
    description=(
        "Merges the separate BLISS scan entries of each raw layer folder into one "
        "darfix-ready .h5 file (detector frames + motor positions). Optional — skip it "
        "if your scans are already concatenated. Writes <folder>_concat.h5 next to each input."
    ),
    params=(
        Param(
            "mode",
            ParamType.ENUM,
            "Mode",
            default="single",
            choices=("single", "batch"),
            help=(
                "single processes one scan folder ('Input folder'); batch processes every "
                "subfolder of 'Root folder' whose name matches 'Folder pattern'."
            ),
        ),
        # --- single mode ---
        Param(
            "input_folder",
            ParamType.DIR,
            "Input folder",
            must_exist=True,
            help="The raw scan folder containing the .h5 file to concatenate (single mode only).",
        ),
        Param(
            "h5_filename_override",
            ParamType.STR,
            "H5 filename override",
            default="",
            advanced=True,
            group="Data layout",
            help=(
                "Name of the .h5 file inside the input folder, if it is not "
                "'<folder name>.h5'. Leave blank to auto-detect (single mode)."
            ),
        ),
        # --- batch mode ---
        Param(
            "root_folder",
            ParamType.DIR,
            "Root folder",
            must_exist=True,
            help=(
                "Parent folder holding one subfolder per layer (batch mode only). Each "
                "matching subfolder is concatenated separately."
            ),
        ),
        Param(
            "folder_pattern",
            ParamType.STR,
            "Folder pattern",
            default="*",
            help=(
                "Glob pattern selecting which subfolders of the root to process in batch mode, "
                "e.g. '*' for all or 'layer_*' for a subset."
            ),
        ),
        Param(
            "skip_existing",
            ParamType.BOOL,
            "Skip existing",
            default=False,
            help=(
                "Skip folders that already contain a _concat.h5 output — useful when "
                "re-running after adding new layers."
            ),
        ),
        # --- shared ---
        Param(
            "entry_suffix",
            ParamType.STR,
            "Entry suffix",
            default=".1",
            advanced=True,
            group="Data layout",
            help=(
                "Only BLISS entries ending in this suffix are merged (e.g. '.1' keeps 1.1, 2.1, …); "
                "other entries such as alignment scans are ignored."
            ),
        ),
        Param(
            "detector_read_path",
            ParamType.STR,
            "Detector read path",
            default="instrument/pco_ff/image",
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path to the detector frames inside each scan entry. Only change if your "
                "beamline files use a different detector or layout."
            ),
        ),
        Param(
            "detector_write_path",
            ParamType.STR,
            "Detector write path",
            default="measurement/pco_ff",
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path where the merged detector data is written inside the output entry "
                "(darfix reads this location)."
            ),
        ),
        Param(
            "positioners_path",
            ParamType.STR,
            "Positioners path",
            default="instrument/positioners",
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path to the motor-position group inside each scan entry; positions are "
                "merged across scans."
            ),
        ),
        Param(
            "output_entry",
            ParamType.STR,
            "Output entry",
            default="entry_0000",
            advanced=True,
            group="Data layout",
            help="Name of the single merged entry in the output file. darfix expects 'entry_0000'.",
        ),
        Param(
            "vds_policy",
            ParamType.ENUM,
            "VDS policy",
            default="relative",
            choices=("relative", "absolute"),
            advanced=True,
            group="Output",
            help=(
                "How the virtual dataset stores references to the source files: relative paths "
                "survive moving the whole tree together; absolute paths break when anything moves. "
                "Ignored when 'Copy data' is on."
            ),
        ),
        Param(
            "copy_data",
            ParamType.BOOL,
            "Copy data",
            default=False,
            advanced=True,
            group="Output",
            help=(
                "Off = write a virtual dataset (fast and small, but it breaks if the source files "
                "move). On = copy the frames into a self-contained archival file (slower, larger)."
            ),
        ),
        Param(
            "overwrite",
            ParamType.BOOL,
            "Overwrite",
            default=True,
            advanced=True,
            group="Output",
            help="Replace an existing output file. If off, folders with an existing output fail instead.",
        ),
    ),
)


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------
@dataclass
class ConcatFileResult:
    """Outcome of concatenating one input file."""

    input_path: str
    output_path: str
    ok: bool = False
    skipped: bool = False
    n_entries: int = 0
    total_frames: int = 0
    n_motors: int = 0
    n_varying: int = 0
    copied: bool = False  # True = data copied, False = VDS
    error: str | None = None


@dataclass
class ConcatResult:
    """Aggregate outcome over one (single) or many (batch) files."""

    files: list[ConcatFileResult] = field(default_factory=list)

    @property
    def outputs(self) -> list[str]:
        return [f.output_path for f in self.files if f.ok]

    @property
    def n_ok(self) -> int:
        return sum(1 for f in self.files if f.ok)

    @property
    def n_skipped(self) -> int:
        return sum(1 for f in self.files if f.skipped)

    @property
    def n_failed(self) -> int:
        return sum(1 for f in self.files if not f.ok and not f.skipped)


# -----------------------------------------------------------------------------
# Core logic (ported)
# -----------------------------------------------------------------------------
def collect_positioners(
    h5f: h5py.File,
    entries: list[str],
    entries_n_frames: list[int],
    positioners_path: str,
) -> dict[str, np.ndarray]:
    """Merge positioners across *entries* (faithful port of the scripts).

    Array motors are concatenated; varying scalars are expanded to one value
    per frame then concatenated; motors uniform across all scans collapse to a
    scalar.
    """
    all_motors: dict[str, list[np.ndarray]] = {}
    for entry, n_frames in zip(entries, entries_n_frames):
        grp_path = f"{entry}/{positioners_path}"
        if grp_path not in h5f:
            continue
        grp = h5f[grp_path]
        for key in grp.keys():
            item = grp[key]
            if not isinstance(item, h5py.Dataset):
                continue
            val = item[()]
            if isinstance(val, np.ndarray) and val.ndim >= 1 and val.size > 1:
                all_motors.setdefault(key, []).append(val.ravel())
            else:
                scalar_val = val.flat[0] if isinstance(val, np.ndarray) else val
                all_motors.setdefault(key, []).append(np.full(n_frames, scalar_val))

    result: dict[str, np.ndarray] = {}
    for key, arrays in all_motors.items():
        concatenated = np.concatenate(arrays)
        try:
            uniques = np.unique(concatenated)
            result[key] = uniques[0] if uniques.size == 1 else concatenated
        except TypeError:
            result[key] = concatenated
    return result


def concatenate_single_file(
    input_path: str,
    output_path: str,
    *,
    entry_suffix: str = ".1",
    detector_read_path: str = "instrument/pco_ff/image",
    detector_write_path: str = "measurement/pco_ff",
    positioners_path: str = "instrument/positioners",
    output_entry: str = "entry_0000",
    vds_policy: str = "relative",
    copy_data: bool = False,
    overwrite: bool = True,
    progress: ProgressFn | None = None,
) -> ConcatFileResult:
    """Concatenate one BLISS ``.h5`` into a darfix-compatible output file."""
    progress = progress or _noop
    res = ConcatFileResult(input_path=input_path, output_path=output_path)

    if os.path.exists(output_path) and not overwrite:
        res.error = "output exists (overwrite disabled)"
        return res

    # Keep the input open across the whole write so both VDS policies and the
    # copy path can read source frames.
    with h5py.File(input_path, "r") as h5in:
        entries = h5io.get_filtered_entries(h5in, entry_suffix)
        if not entries:
            res.error = f"no entries ending with {entry_suffix!r}"
            return res

        # --- detector scan: gather valid entries, counts, frame geometry ---
        valid_entries: list[str] = []
        entries_n_frames: list[int] = []
        frame_shape: tuple[int, ...] | None = None
        frame_dtype: np.dtype | None = None
        for i, entry in enumerate(entries):
            det_path = f"{entry}/{detector_read_path}"
            if det_path not in h5in:
                continue
            ds = h5in[det_path]
            if ds.ndim != 3:
                continue
            n_fr, fshape, dtype = h5io.detector_info(ds)
            if frame_shape is None:
                frame_shape, frame_dtype = fshape, dtype
            elif fshape != frame_shape:
                res.error = f"frame shape mismatch in {entry}: {fshape} vs {frame_shape}"
                return res
            valid_entries.append(entry)
            entries_n_frames.append(n_fr)
            progress(0.4 * (i + 1) / len(entries), f"scan {entry}: {n_fr} frames")

        if not entries_n_frames:
            res.error = "no valid 3-D detector data found"
            return res

        total_frames = int(sum(entries_n_frames))

        # --- positioners (uses the full filtered entry list, like the scripts) ---
        positioners = collect_positioners(h5in, entries, entries_n_frames, positioners_path)
        n_varying = sum(
            1 for v in positioners.values() if isinstance(v, np.ndarray) and v.ndim >= 1
        )

        # --- write output ---
        if os.path.exists(output_path) and overwrite:
            os.remove(output_path)

        out_det_path = f"{output_entry}/{detector_write_path}"
        with h5py.File(output_path, "w") as h5out:
            entry_grp = h5out.create_group(output_entry)
            entry_grp.require_group("measurement")

            if copy_data:
                progress(0.5, "copying detector frames")
                dset = h5out.create_dataset(
                    out_det_path,
                    shape=(total_frames, *frame_shape),
                    dtype=frame_dtype,
                    compression="gzip",
                )
                idx = 0
                for j, entry in enumerate(valid_entries):
                    src = h5in[f"{entry}/{detector_read_path}"]
                    n = entries_n_frames[j]
                    dset[idx : idx + n] = src[()]
                    idx += n
                    progress(0.5 + 0.4 * (j + 1) / len(valid_entries), f"copied {entry}")
            else:
                progress(0.5, "building detector VDS")
                sources = [
                    h5io.make_virtual_source(
                        h5in[f"{entry}/{detector_read_path}"], output_path, vds_policy
                    )
                    for entry in valid_entries
                ]
                layout = h5io.build_virtual_layout(
                    sources, entries_n_frames, frame_shape, frame_dtype
                )
                h5out.create_virtual_dataset(out_det_path, layout)

            # positioners
            pos_grp = entry_grp.create_group("instrument/positioners")
            for motor_name, value in sorted(positioners.items()):
                pos_grp.create_dataset(motor_name, data=value)

            # metadata
            n_entries = len(entries)
            entry_grp.attrs["num_scans"] = n_entries
            entry_grp.attrs["source_entries"] = ", ".join(entries)
            entry_grp.attrs["source_file"] = os.path.basename(input_path)
            entry_grp.attrs["description"] = (
                f"Concatenation of {n_entries} scans (entries ending in {entry_suffix!r})"
            )

    progress(1.0, "done")
    res.ok = True
    res.n_entries = len(entries)
    res.total_frames = total_frames
    res.n_motors = len(positioners)
    res.n_varying = n_varying
    res.copied = copy_data
    return res


# -----------------------------------------------------------------------------
# Stage entry point
# -----------------------------------------------------------------------------
def run(params: dict, progress: ProgressFn | None = None) -> ConcatResult:
    """Run the concat stage in single or batch mode.

    *params* is a flat dict following :data:`STAGE` (missing keys fall back to
    the schema defaults). *progress* receives ``(fraction, message)``.
    """
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}

    common = dict(
        entry_suffix=p["entry_suffix"],
        detector_read_path=p["detector_read_path"],
        detector_write_path=p["detector_write_path"],
        positioners_path=p["positioners_path"],
        output_entry=p["output_entry"],
        vds_policy=p["vds_policy"],
        copy_data=bool(p["copy_data"]),
        overwrite=bool(p["overwrite"]),
    )

    if p["mode"] == "single":
        folder = p["input_folder"]
        if not folder:
            raise ValueError("single mode requires 'input_folder'")
        override = p.get("h5_filename_override") or None
        input_path = h5io.resolve_input_file(folder, override)
        output_path = h5io.make_output_path(input_path)
        progress(0.0, f"concat {os.path.basename(input_path)}")
        fr = concatenate_single_file(input_path, output_path, progress=progress, **common)
        return ConcatResult(files=[fr])

    if p["mode"] == "batch":
        root = (p["root_folder"] or "").rstrip("/")
        if not root:
            raise ValueError("batch mode requires 'root_folder'")
        folders = find_matching_folders(root, p["folder_pattern"])
        if not folders:
            raise ValueError(f"no folders matching {p['folder_pattern']!r} in {root}")
        skip_existing = bool(p["skip_existing"])
        result = ConcatResult()
        for i, folder in enumerate(folders):
            input_path = h5io.resolve_input_file(folder)
            output_path = h5io.make_output_path(input_path)
            base = os.path.basename(folder)
            frac0 = i / len(folders)

            def sub(local: float, msg: str, frac0=frac0, base=base) -> None:
                progress(frac0 + local / len(folders), f"[{i + 1}/{len(folders)}] {base}: {msg}")

            if not os.path.exists(input_path):
                result.files.append(
                    ConcatFileResult(input_path, output_path, error="input .h5 not found")
                )
                continue
            if skip_existing and os.path.exists(output_path):
                result.files.append(
                    ConcatFileResult(
                        input_path,
                        output_path,
                        skipped=True,
                        error="output exists (skip_existing)",
                    )
                )
                sub(1.0, "skipped (already done)")
                continue
            try:
                fr = concatenate_single_file(input_path, output_path, progress=sub, **common)
            except Exception as exc:  # noqa: BLE001 - report per-file, keep batch going
                fr = ConcatFileResult(input_path, output_path, error=str(exc))
            result.files.append(fr)
        progress(
            1.0,
            f"batch done: {result.n_ok} ok, {result.n_skipped} skipped, {result.n_failed} failed",
        )
        return result

    raise ValueError(f"unknown mode {p['mode']!r} (expected 'single' or 'batch')")


# -----------------------------------------------------------------------------
# Headless CLI
# -----------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Concatenate BLISS scans (darfix-compatible).")
    ap.add_argument("--mode", choices=("single", "batch"), default="single")
    ap.add_argument("--input-folder", default="", help="single mode: folder holding the .h5")
    ap.add_argument("--h5-filename-override", default="")
    ap.add_argument("--root-folder", default="", help="batch mode: parent of layer folders")
    ap.add_argument("--folder-pattern", default="*")
    ap.add_argument("--entry-suffix", default=".1")
    ap.add_argument("--vds-policy", choices=("relative", "absolute"), default="relative")
    ap.add_argument("--copy-data", action="store_true", help="self-contained copy instead of VDS")
    ap.add_argument("--no-overwrite", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)

    params = dict(
        mode=args.mode,
        input_folder=args.input_folder,
        h5_filename_override=args.h5_filename_override,
        root_folder=args.root_folder,
        folder_pattern=args.folder_pattern,
        entry_suffix=args.entry_suffix,
        vds_policy=args.vds_policy,
        copy_data=args.copy_data,
        overwrite=not args.no_overwrite,
        skip_existing=args.skip_existing,
    )

    def show(frac: float, msg: str) -> None:
        print(f"  [{frac * 100:5.1f}%] {msg}")

    result = run(params, progress=show)
    print(f"\n{result.n_ok} ok, {result.n_skipped} skipped, {result.n_failed} failed")
    for fr in result.files:
        status = "OK" if fr.ok else ("SKIP" if fr.skipped else "FAIL")
        detail = (
            f"{fr.n_entries} scans, {fr.total_frames} frames, "
            f"{fr.n_motors} motors ({fr.n_varying} varying), "
            f"{'copy' if fr.copied else 'vds'}"
            if fr.ok
            else (fr.error or "")
        )
        print(f"  [{status}] {os.path.basename(fr.output_path)}  {detail}")
    return 0 if result.n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main())
