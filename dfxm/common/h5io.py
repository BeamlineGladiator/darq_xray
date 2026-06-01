"""Shared HDF5 I/O primitives for the DFXM pipeline.

These are the reusable, stage-agnostic pieces: locating the input ``.h5``,
filtering BLISS entries, building detector virtual datasets (VDS), reading
positioner motors, and validating a darfix ``maps.h5`` before the downstream
stages consume it. Stage-specific orchestration (e.g. the concat positioner
expand/collapse rules) lives in the stage modules, not here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import h5py
import numpy as np

from .sort import natural_sort_key


# -----------------------------------------------------------------------------
# Path conventions
# -----------------------------------------------------------------------------
def resolve_input_file(folder: str, override: str | None = None) -> str:
    """H5 file path for *folder*.

    Convention (ESRF/BLISS): the file shares its folder's name, e.g.
    ``scan_26/scan_26.h5``. *override* forces a specific filename inside the
    folder instead of auto-detecting.
    """
    if override:
        return os.path.join(folder, override)
    folder_name = os.path.basename(folder.rstrip("/"))
    return os.path.join(folder, folder_name + ".h5")


def make_output_path(input_path: str, suffix: str = "_concat") -> str:
    """Output path next to *input_path*: ``<stem><suffix><ext>``."""
    root, ext = os.path.splitext(input_path)
    return root + suffix + ext


def get_filtered_entries(h5f: h5py.File, entry_suffix: str) -> list[str]:
    """Top-level groups whose name ends with *entry_suffix*, natural-sorted.

    BLISS writes scans as ``1.1``, ``2.1``, …; ``entry_suffix=".1"`` keeps the
    primary scan of each acquisition.
    """
    entries = [k for k in h5f.keys() if isinstance(h5f[k], h5py.Group) and k.endswith(entry_suffix)]
    entries.sort(key=natural_sort_key)
    return entries


# -----------------------------------------------------------------------------
# Virtual datasets (VDS)
# -----------------------------------------------------------------------------
def make_virtual_source(
    dataset: h5py.Dataset, output_file: str, policy: str = "relative"
) -> h5py.VirtualSource:
    """A :class:`h5py.VirtualSource` for *dataset*, referencing it by path.

    ``policy="relative"`` (default) stores a path relative to *output_file* so
    the combined file stays portable as long as the originals keep their
    relative layout; ``policy="absolute"`` embeds the absolute source path.
    """
    if policy == "absolute":
        return h5py.VirtualSource(dataset)
    if policy != "relative":
        raise ValueError(f"vds policy must be 'relative' or 'absolute', got {policy!r}")
    relpath = os.path.relpath(
        os.path.abspath(dataset.file.filename),
        os.path.dirname(os.path.abspath(output_file)),
    )
    if not relpath.startswith("./") and not relpath.startswith("../"):
        relpath = "./" + relpath
    return h5py.VirtualSource(
        path_or_dataset=relpath,
        name=dataset.name,
        shape=dataset.shape,
        dtype=dataset.dtype,
    )


def build_virtual_layout(
    sources: list[h5py.VirtualSource],
    frame_counts: list[int],
    frame_shape: tuple[int, ...],
    dtype: np.dtype,
) -> h5py.VirtualLayout:
    """Stack per-scan detector *sources* along axis 0 into one VirtualLayout."""
    total = int(sum(frame_counts))
    layout = h5py.VirtualLayout(shape=(total, *frame_shape), dtype=dtype)
    idx = 0
    for n, src in zip(frame_counts, sources):
        layout[idx : idx + n] = src
        idx += n
    return layout


# -----------------------------------------------------------------------------
# Readers
# -----------------------------------------------------------------------------
def detector_info(dataset: h5py.Dataset) -> tuple[int, tuple[int, ...], np.dtype]:
    """``(n_frames, frame_shape, dtype)`` for a 3-D detector stack."""
    if dataset.ndim != 3:
        raise ValueError(f"expected a 3-D detector dataset, got {dataset.ndim}-D: {dataset.name}")
    return dataset.shape[0], tuple(dataset.shape[1:]), dataset.dtype


def read_positioners(h5f: h5py.File, group_path: str) -> dict[str, np.ndarray]:
    """Read every dataset under *group_path* as-is into a name -> value dict.

    Values are returned exactly as stored (scalars stay scalars, arrays stay
    arrays). This is the raw reader; the concat stage applies its own
    expand/collapse rules on top.
    """
    if group_path not in h5f:
        raise KeyError(f"positioners group not found: {group_path}")
    grp = h5f[group_path]
    out: dict[str, np.ndarray] = {}
    for key in grp.keys():
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            out[key] = item[()]
    return out


def read_samy_samz(
    h5f: h5py.File,
    group_path: str,
    samy_key: str = "samy",
    samz_key: str = "samz",
) -> tuple[np.ndarray, np.ndarray]:
    """Read the ``samy`` and ``samz`` sample-translation stages from *group_path*."""
    pos = read_positioners(h5f, group_path)
    missing = [k for k in (samy_key, samz_key) if k not in pos]
    if missing:
        raise KeyError(f"missing positioners {missing} in {group_path}")
    return np.asarray(pos[samy_key]), np.asarray(pos[samz_key])


# -----------------------------------------------------------------------------
# darfix maps.h5 validation
# -----------------------------------------------------------------------------
@dataclass
class MapsValidation:
    """Result of validating a darfix ``maps.h5`` file."""

    path: str
    ok: bool
    present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    error: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def validate_maps_file(path: str, required_paths: list[str]) -> MapsValidation:
    """Check that *path* exists and contains every dataset in *required_paths*.

    Used to bracket darfix: a ``maps.h5`` must carry the COM datasets the
    strain/mosaicity stages read before those stages are allowed to run.
    Never raises for a bad file — returns a :class:`MapsValidation` describing
    what is present/missing.
    """
    if not os.path.exists(path):
        return MapsValidation(
            path=path, ok=False, missing=list(required_paths), error="file not found"
        )
    present: list[str] = []
    missing: list[str] = []
    shapes: dict[str, tuple[int, ...]] = {}
    try:
        with h5py.File(path, "r") as h5f:
            for p in required_paths:
                obj = h5f.get(p)
                if isinstance(obj, h5py.Dataset):
                    present.append(p)
                    shapes[p] = tuple(obj.shape)
                else:
                    missing.append(p)
    except OSError as exc:
        return MapsValidation(path=path, ok=False, missing=list(required_paths), error=str(exc))
    return MapsValidation(
        path=path,
        ok=not missing,
        present=present,
        missing=missing,
        shapes=shapes,
    )
