"""Detect experiment settings from the data on disk.

Qt-free. Each detector inspects the raw/processed trees and returns
:class:`Detection` rows — never raising for "not there yet": a missing
input becomes a row with ``error`` set (the same skip-with-reason style the
stages use), so one unreadable file cannot block the other detections.

The orchestrator :func:`detect_experiment` runs every detector against an
:class:`~dfxm.config.models.Experiment` (typically the live form values) and
is re-runnable: a first pass before darfix fills the raw-data facts
(patterns, entry suffix, pixel sizes, ccmth positioner fallback); run again
after darfix and the maps.h5-derived rows (ccmth COM median, darfix-ROI
size) appear. darfix records no ROI metadata, so only the crop *size* is
recoverable — the origin must still be typed from the darfix widget.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import h5py

_FAMILY_RE = re.compile(r"^(?P<stem>.+)__(?P<num>\d+)$")
_ENTRY_RE = re.compile(r"^\d+(\.\d+)$")


@dataclass(frozen=True)
class Detection:
    """One detected experiment value (or the reason there isn't one).

    ``value is None and error is None`` marks an info-only row (e.g.
    "darfix ROI size consistent") — nothing to apply, nothing wrong.
    """

    field: str
    value: Any | None = None
    note: str = ""
    error: str | None = None


# -- folder patterns ----------------------------------------------------------


def folder_families(raw_root: str) -> dict[str, int]:
    """Group ``<stem>__<N>`` subfolders of *raw_root*: stem -> folder count."""
    try:
        names = sorted(os.listdir(raw_root))
    except OSError:
        return {}
    fams: Counter[str] = Counter()
    for n in names:
        m = _FAMILY_RE.match(n)
        if m and os.path.isdir(os.path.join(raw_root, n)):
            fams[m.group("stem")] += 1
    return dict(fams)


def detect_patterns(raw_root: str) -> list[Detection]:
    """Suggest folder/mosa/rocking glob patterns from the folder families."""
    fams = folder_families(raw_root)
    if not fams:
        return [
            Detection(
                "folder_pattern",
                error=f"no '<name>__<N>' folder families under {raw_root or '(unset)'}",
            )
        ]
    out: list[Detection] = []
    remaining = dict(fams)

    def take(field: str, key: str) -> None:
        match = {s: c for s, c in remaining.items() if key in s.lower()}
        if not match:
            out.append(Detection(field, error=f"no folder family containing {key!r}"))
            return
        stem = max(match, key=match.__getitem__)  # largest family wins
        del remaining[stem]
        out.append(Detection(field, f"{stem}__*", f"{fams[stem]} folders"))

    take("mosa_pattern", "mosa")
    take("rocking_pattern", "rocking")
    if remaining:
        stem = max(remaining, key=remaining.__getitem__)
        out.insert(0, Detection("folder_pattern", f"{stem}__*", f"{fams[stem]} folders"))
    else:
        out.insert(0, Detection("folder_pattern", error="no folder family besides mosa/rocking"))
    return out


# -- scan file + entry suffix -------------------------------------------------


def select_scan_file(folder: str) -> str | None:
    """The raw scan .h5 inside a layer *folder* (concat output excluded).

    Prefers ``<folder name>.h5`` exactly (the BLISS convention); otherwise the
    first ``*.h5`` that is not a ``*_concat.h5``.
    """
    preferred = os.path.join(folder, os.path.basename(os.path.normpath(folder)) + ".h5")
    if os.path.isfile(preferred):
        return preferred
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return None
    for n in names:
        if n.endswith(".h5") and not n.endswith("_concat.h5"):
            return os.path.join(folder, n)
    return None


def detect_entry_suffix(scan_h5: str) -> Detection:
    """Majority BLISS entry suffix (``1.1`` -> ``.1``) among *scan_h5* entries."""
    try:
        with h5py.File(scan_h5, "r") as f:
            suffixes = Counter(m.group(1) for k in f.keys() if (m := _ENTRY_RE.match(k)))
    except OSError as exc:
        return Detection("entry_suffix", error=f"could not read {scan_h5}: {exc}")
    if not suffixes:
        return Detection(
            "entry_suffix", error=f"no '<n>.<m>' entries in {os.path.basename(scan_h5)}"
        )
    suffix, count = suffixes.most_common(1)[0]
    note = f"{count} entries in {os.path.basename(scan_h5)}"
    if len(suffixes) > 1:
        others = ", ".join(s for s in sorted(suffixes) if s != suffix)
        note += f" (mixed with {others} — majority wins)"
    return Detection("entry_suffix", suffix, note)


# -- calibration --------------------------------------------------------------


def detect_pixel_sizes(scan_h5: str, positioners_path: str, entry_suffix: str) -> list[Detection]:
    """Pixel sizes X/Y via :func:`~dfxm.common.pixel_size.compute_pixel_size`.

    Two rows sharing one geometry note; on failure both rows carry the same
    error so the review table stays honest about what is missing.
    """
    from dfxm.common.errors import StageUserError
    from dfxm.common.pixel_size import compute_pixel_size

    fields = ("pixel_size_x_um", "pixel_size_y_um")
    try:
        res = compute_pixel_size(
            scan_h5, positioners_path=positioners_path, entry_suffix=entry_suffix
        )
    except StageUserError as exc:
        err = f"{exc} — {exc.hint}" if exc.hint else str(exc)
        return [Detection(f, error=err) for f in fields]
    except Exception as exc:  # noqa: BLE001 — unreadable/foreign file must not block others
        return [
            Detection(f, error=f"could not read {os.path.basename(scan_h5)}: {exc}") for f in fields
        ]
    note = (
        f"M={res.magnification:.3f}, {res.objective} objective, "
        f"2θ={res.two_theta_deg:.3f}°, condenser {'in' if res.condenser_in else 'out'} — "
        f"{os.path.basename(scan_h5)}"
    )
    return [
        Detection("pixel_size_x_um", round(res.pixel_size_x_um, 6), note),
        Detection("pixel_size_y_um", round(res.pixel_size_y_um, 6), note),
    ]


def find_strain_maps(
    processed_root: str, pattern: str, maps_filename: str, ccmth_com_path: str
) -> tuple[str, str] | None:
    """First layer folder under *processed_root* whose maps file has the ccmth COM.

    Returns ``(maps_path, folder_name)`` or None. Mosa-family maps carry chi/mu
    only, so every candidate is probed for *ccmth_com_path* before acceptance.
    """
    from dfxm.common.sort import find_matching_folders

    if not processed_root or not os.path.isdir(processed_root):
        return None
    for folder in find_matching_folders(processed_root, pattern):
        maps_path = os.path.join(folder, maps_filename)
        if not os.path.isfile(maps_path):
            continue
        try:
            with h5py.File(maps_path, "r") as f:
                if ccmth_com_path in f:
                    return maps_path, os.path.basename(folder)
        except OSError:
            continue
    return None


def detect_ccmth_from_maps(maps_path: str, folder_name: str, ccmth_com_path: str) -> Detection:
    """ccmth reference suggestion: nanmedian of the darfix ccmth COM map."""
    import numpy as np

    try:
        with h5py.File(maps_path, "r") as f:
            com = f[ccmth_com_path][()]
        value = float(np.nanmedian(com))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return Detection(
            "ccmth_ref_deg", error=f"could not read ccmth COM from {folder_name}: {exc}"
        )
    return Detection("ccmth_ref_deg", round(value, 4), f"median of ccmth COM map, {folder_name}")


def detect_ccmth_from_positioners(
    scan_h5: str, positioners_path: str, entry_suffix: str
) -> Detection:
    """Fallback ccmth reference from the raw scan's motor snapshot (pre-darfix)."""
    import numpy as np

    from dfxm.common.h5io import get_filtered_entries, read_positioners

    try:
        with h5py.File(scan_h5, "r") as f:
            entries = get_filtered_entries(f, entry_suffix)
            if not entries:
                return Detection(
                    "ccmth_ref_deg",
                    error=f"no {entry_suffix!r} entries in {os.path.basename(scan_h5)}",
                )
            pos = read_positioners(f, f"{entries[0]}/{positioners_path}")
    except (OSError, KeyError) as exc:
        return Detection("ccmth_ref_deg", error=f"could not read positioners: {exc}")
    if "ccmth" not in pos:
        return Detection("ccmth_ref_deg", error=f"no 'ccmth' motor in {os.path.basename(scan_h5)}")
    value = float(np.asarray(pos["ccmth"]).reshape(-1)[0])
    return Detection(
        "ccmth_ref_deg",
        round(value, 4),
        "single motor snapshot — confirm against the beamline alignment",
    )


def detect_darfix_roi(
    maps_path: str, folder_name: str, ccmth_com_path: str, current_roi: str
) -> Detection:
    """Darfix-ROI row from the map shape: size only — darfix records no origin.

    Blank/malformed current -> an applicable-but-partial ``?,?,w,h`` value (the
    GUI keeps it uncheckable until the origin is typed). Filled current ->
    validation: matching size is an info row; a mismatch offers the corrected
    size with the existing origin kept.
    """
    from dfxm.common.roi import parse_darfix_roi

    try:
        with h5py.File(maps_path, "r") as f:
            h, w = f[ccmth_com_path].shape[:2]
    except (OSError, KeyError, ValueError) as exc:
        return Detection("darfix_roi", error=f"could not read map shape from {folder_name}: {exc}")
    try:
        win = parse_darfix_roi(current_roi)
    except ValueError:
        win = None
    if win is None:
        return Detection(
            "darfix_roi",
            f"?,?,{w},{h}",
            f"map size {w}×{h} from {folder_name} — replace ?,? with the darfix origin",
        )
    if (win.width, win.height) == (w, h):
        return Detection("darfix_roi", None, f"✓ size matches maps.h5 ({w}×{h})")
    return Detection(
        "darfix_roi",
        f"{win.origin_x},{win.origin_y},{w},{h}",
        f"size in maps.h5 is {w}×{h}, not {win.width}×{win.height} — origin kept",
    )


# -- orchestrator -------------------------------------------------------------


def detect_experiment(current) -> list[Detection]:
    """Run every detector against *current* (an :class:`Experiment`).

    Re-runnable: rows for data that does not exist yet come back as
    skip-with-reason, so a pre-darfix pass already shows what a later pass
    will add. Never overwrites anything — callers decide what to apply.
    """
    from dfxm.common.sort import find_matching_folders

    raw_root = (current.raw_root or "").rstrip("/")
    if not raw_root or not os.path.isdir(raw_root):
        return [Detection("raw_root", error="set Raw data root to an existing folder first")]

    out = detect_patterns(raw_root)
    detected = {d.field: d.value for d in out if d.value}
    pattern = (
        current.folder_pattern
        if current.folder_pattern not in ("", "*")
        else detected.get("folder_pattern", "")
    )

    scan = None
    if pattern:
        folders = find_matching_folders(raw_root, pattern)
        scan = select_scan_file(folders[0]) if folders else None
    suffix = current.entry_suffix or ".1"
    if scan is None:
        if pattern:
            out.append(
                Detection(
                    "entry_suffix", error=f"no layer folder matching {pattern!r} has a scan .h5"
                )
            )
            out.extend(
                Detection(f, error="pixel sizes need a raw scan — none found")
                for f in ("pixel_size_x_um", "pixel_size_y_um")
            )
        else:
            no_pattern = "no folder pattern — detect or set Folder pattern first"
            out.append(Detection("entry_suffix", error=no_pattern))
            out.extend(
                Detection(f, error=no_pattern) for f in ("pixel_size_x_um", "pixel_size_y_um")
            )
    else:
        suffix_row = detect_entry_suffix(scan)
        out.append(suffix_row)
        suffix = suffix_row.value or suffix
        out.extend(detect_pixel_sizes(scan, current.positioners_path, suffix))

    found = (
        find_strain_maps(
            current.processed_root, pattern, current.maps_filename, current.ccmth_com_path
        )
        if pattern
        else None
    )
    if found:
        maps_path, folder_name = found
        out.append(detect_ccmth_from_maps(maps_path, folder_name, current.ccmth_com_path))
        out.append(
            detect_darfix_roi(maps_path, folder_name, current.ccmth_com_path, current.darfix_roi)
        )
    else:
        if scan is not None:
            out.append(detect_ccmth_from_positioners(scan, current.positioners_path, suffix))
        if pattern:
            darfix_error = "no darfix maps.h5 under the processed root yet — re-run after darfix"
        else:
            darfix_error = (
                "no folder pattern to locate maps.h5 — detect or set Folder pattern first"
            )
        out.append(Detection("darfix_roi", error=darfix_error))
    return out


# -- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Print the detection table for a raw (and optionally processed) tree."""
    import argparse

    from dfxm.config.models import Experiment

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("raw_root", help="RAW_DATA root (the folder holding the layer subfolders)")
    ap.add_argument(
        "--processed-root", default="", help="PROCESSED_DATA root (enables the maps.h5 rows)"
    )
    ap.add_argument("--maps-filename", default="maps.h5")
    args = ap.parse_args(argv)
    exp = Experiment(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        maps_filename=args.maps_filename,
    )
    for d in detect_experiment(exp):
        if d.error:
            print(f"{d.field:18} SKIP  {d.error}")
        elif d.value is None:
            print(f"{d.field:18} INFO  {d.note}")
        else:
            print(f"{d.field:18} {d.value!s:26} {d.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
