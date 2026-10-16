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
from collections.abc import Iterable


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


def drop_plot_dirs(
    folders: list[str],
    *,
    plot_dirs: Iterable[str | None],
    maps_filename: str,
    markers: tuple[str, ...],
) -> list[str]:
    """*folders* without a stage's own diagnostics folders.

    A stage that writes its PNGs into a folder inside the root — strain's
    ``strain_maps``, mosaicity's ``mosaicity_detrend`` — has that folder
    matched by a broad folder pattern (both stages default to ``*``) on every
    later run: it was reported as a skipped layer, counted in the estimate's
    layer total, and, when it sorted first, sized as ``work[0]``.

    A folder holding *maps_filename* is a layer and is always kept, even when
    the output folder points at it. Of the rest, dropped: every folder in
    *plot_dirs* (the default and the configured output folder; blank entries
    are ignored), and any folder holding a file ending in one of *markers* —
    that last one is how a folder an EARLIER run wrote to is recognised after
    the output folder has been changed. A folder with neither is kept, so the
    stage still reports a layer whose maps file is missing.
    """
    named = {os.path.realpath(d) for d in plot_dirs if d}

    def is_plot_dir(folder: str) -> bool:
        if os.path.exists(os.path.join(folder, maps_filename)):
            return False
        if os.path.realpath(folder) in named:
            return True
        try:
            return any(n.endswith(markers) for n in os.listdir(folder))
        except OSError:
            return False

    return [f for f in folders if not is_plot_dir(f)]
