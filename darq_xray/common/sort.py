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
    (single -> ``input_folder``; batch -> ``root_folder`` + ``folder_pattern``,
    minus :func:`drop_output_dirs`) so an estimator predicts the same job the
    run will do. Returns [] rather than raising when nothing resolves —
    estimators are advisory.
    """
    if params.get("mode") == "single":
        folder = str(params.get("input_folder") or "")
        folders = [folder] if folder else []
    else:
        root = str(params.get("root_folder") or "").rstrip("/")
        folders = find_matching_folders(root, params.get("folder_pattern") or "*") if root else []
        folders = drop_output_dirs(folders, {**params, "maps_filename": maps_filename}, root)
    return [os.path.join(f, maps_filename) for f in folders if f]


# Folders the app itself writes into the processed root — the root strain and
# mosaicity search for layer folders, with patterns that default to "*". The
# stages and dialogs that create them take the names from here, so a rename
# cannot silently re-open the folder to the layer search.
STRAIN_PLOTS_DIR = "strain_maps"
MOSAICITY_DETREND_DIR = "mosaicity_detrend"
REPLOTS_DIR = "replots"
VISUALIZE_DIR = "aligned_volume_visualizations"
SLICES_DIR = "oblique_slices"
PARAVIEW_DIR = "paraview_exports"
APP_OUTPUT_DIRS = (
    STRAIN_PLOTS_DIR,
    MOSAICITY_DETREND_DIR,
    REPLOTS_DIR,
    VISUALIZE_DIR,
    SLICES_DIR,
    PARAVIEW_DIR,
)

# Filename endings of the per-layer PNGs strain and mosaicity write.
PLOT_MARKERS = ("_strain.png", "_hist.png", "_detrend_diag.png")


def drop_output_dirs(folders: list[str], params: dict, root: str) -> list[str]:
    """*folders* without the folders the app writes into *root*.

    Without this a broad folder pattern (strain's and mosaicity's default is
    ``*``) matched them on every later run: each was reported as a layer
    without a maps file, counted in the estimate's layer total, and, when it
    sorted first (``aligned_volume_visualizations`` does), sized as
    ``work[0]`` by the estimate and searched by strain's ROI preview.

    A folder holding the maps file (``params["maps_filename"]``) is a layer
    and is always kept, even when ``params["output_dir"]`` points at it. Of
    the rest, dropped: every :data:`APP_OUTPUT_DIRS` folder under *root* and
    the configured ``output_dir``, by name and even while empty; and any
    folder holding a file ending in one of :data:`PLOT_MARKERS` — how a folder
    an EARLIER run wrote to is recognised after ``output_dir`` has been
    changed. A folder with neither is kept, so the stage still reports a
    layer whose maps file is missing.
    """
    named = {os.path.realpath(os.path.join(root, d)) for d in APP_OUTPUT_DIRS}
    if params.get("output_dir"):
        named.add(os.path.realpath(params["output_dir"]))
    maps_filename = str(params.get("maps_filename") or "maps.h5")

    def is_output_dir(folder: str) -> bool:
        if os.path.exists(os.path.join(folder, maps_filename)):
            return False
        if os.path.realpath(folder) in named:
            return True
        try:
            return any(n.endswith(PLOT_MARKERS) for n in os.listdir(folder))
        except OSError:
            return False

    return [f for f in folders if not is_output_dir(f)]
