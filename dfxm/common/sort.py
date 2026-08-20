"""Natural sorting and folder discovery.

Extracted verbatim (in behaviour) from the helpers duplicated across the
legacy scripts (``concatenate_h5_scans_v3``, ``batch_concatenate_h5_scans_v1``,
``y_calc_axial_strain_v6_batch``, …) so every stage sorts layers identically:
``layer__2`` before ``layer__10``.
"""

from __future__ import annotations

import glob
import os
import re


def natural_sort_key(s: str) -> list:
    """Sort key that orders embedded numbers numerically, not lexically.

    ``"x__10"`` sorts after ``"x__2"`` because the digit runs are compared as
    integers. Non-digit runs are lower-cased for case-insensitive ordering.
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_matching_folders(root_folder: str, pattern: str) -> list[str]:
    """Directories under *root_folder* matching the glob *pattern*, natural-sorted.

    Only directories are returned (matching files are ignored). Sorting is by
    the basename so per-layer folders come back in acquisition order.
    """
    search = os.path.join(root_folder, pattern)
    folders = [f for f in glob.glob(search) if os.path.isdir(f)]
    folders.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    return folders


def resolve_layer_work(params: dict, *, maps_filename: str) -> list[str]:
    """The ``<folder>/<maps_filename>`` paths a folder-based stage would process.

    Mirrors the work-list resolution in ``strain.run`` / ``mosaicity.run``
    (single -> ``input_folder``; batch -> ``root_folder`` + ``folder_pattern``)
    so an estimator predicts the same job the run will do. Returns [] rather
    than raising when nothing resolves — estimators are advisory.
    """
    if params.get("mode") == "single":
        folder = str(params.get("input_folder") or "")
        folders = [folder] if folder else []
    else:
        root = str(params.get("root_folder") or "").rstrip("/")
        folders = find_matching_folders(root, params.get("folder_pattern") or "*") if root else []
    return [os.path.join(f, maps_filename) for f in folders if f]
