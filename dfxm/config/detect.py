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
