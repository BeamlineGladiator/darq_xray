"""Shared HDF5 I/O primitives for the DARQ DFXM pipeline.

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


def sum_dataset_bytes(path: str) -> tuple[int, tuple[int, ...] | None, int]:
    """Total in-memory size of every dataset in *path*, from shapes alone.

    Returns ``(total_bytes, largest_shape, largest_itemsize)``. Walks nested
    groups, reads no data, and returns ``(0, None, 0)`` for anything it cannot
    open — sizing is advisory and must never raise.
    """
    total = 0
    largest_elems = 0
    largest_shape: tuple[int, ...] | None = None
    largest_itemsize = 0

    def visit(_name, obj):
        nonlocal total, largest_elems, largest_shape, largest_itemsize
        if not isinstance(obj, h5py.Dataset):
            return
        n = 1
        for dim in obj.shape:
            n *= int(dim)
        itemsize = int(obj.dtype.itemsize)
        total += n * itemsize
        if n > largest_elems:
            largest_elems = n
            largest_shape = tuple(int(d) for d in obj.shape)
            largest_itemsize = itemsize

    try:
        with h5py.File(path, "r") as f:
            f.visititems(visit)
    except Exception:  # noqa: BLE001 - unreadable input -> unknown size
        return 0, None, 0
    return total, largest_shape, largest_itemsize


def iter_dataset_sizes(path: str) -> list[tuple[str, tuple[int, ...], int]]:
    """``(name, shape, itemsize)`` for every dataset in *path*, from shapes alone.

    Same traversal as :func:`sum_dataset_bytes` (``f.visititems``, reads no
    data) but returns one entry per dataset instead of a running total, so a
    caller can size the *specific* dataset it is about to load rather than the
    whole file. ``name`` matches the in-file path (e.g. ``"chi/Center of
    mass"``). Returns ``[]`` for anything unreadable — sizing is advisory and
    must never raise.
    """
    out: list[tuple[str, tuple[int, ...], int]] = []

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        shape = tuple(int(d) for d in obj.shape)
        out.append((name, shape, int(obj.dtype.itemsize)))

    try:
        with h5py.File(path, "r") as f:
            f.visititems(visit)
    except Exception:  # noqa: BLE001 - unreadable input -> unknown size
        return []
    return out


# -----------------------------------------------------------------------------
# Incremental volume writing
# -----------------------------------------------------------------------------
class StackedVolumeFile:
    """Build a (Z, Y, X) volume file one layer at a time.

    ``strain`` and ``mosaicity`` used to collect every layer in a list and
    ``np.stack`` it, which costs two whole volumes for a product that is
    written once and never re-read. Appending into a resizable dataset costs
    one layer.

    Writes to ``<path>.part`` and renames on a clean close, so a failure
    mid-run leaves nothing behind — the same all-or-nothing behaviour the
    write-at-the-end version had for free.

    The part file is opened **lazily, on the first :meth:`append`**, not in
    ``__init__``. A stage can therefore construct the writer before its layer
    loop (which is what keeps the loop's peak at one layer) without touching
    the filesystem for a run that turns out to produce no layers — e.g. single
    mode pointed at a folder that does not exist, which must stay a
    ``result.skipped`` entry rather than becoming a raw ``FileNotFoundError``.

    **Constructing** the writer unlinks any ``<path>.part`` left over from an
    earlier run. This pipeline runs one stage at a time (``StageRunner`` spawns
    a single child per run and the GUI serialises them), so a part file already
    sitting there is by definition the orphan of a run that was cancelled — the
    runner SIGTERM/SIGKILLs the child, so nothing gets the chance to clean up —
    and for a real volume that orphan is gigabytes. Reclaiming it at
    construction rather than at open means it happens even for a re-run that
    goes on to produce no layers, which never opens anything. Deleting the
    orphan and then failing leaves the user with neither file; that is
    deliberate, since a truncated ``.part`` is unusable either way.

    Construction still creates nothing — no file, and no directory for a
    ``path`` whose parent does not exist.
    """

    def __init__(self, path: str, *, compression: str | None = "gzip") -> None:
        self._path = path
        self._part = path + ".part"
        self._compression = compression
        self._shapes: dict[str, set[tuple[int, ...]]] = {}
        self._closed = False
        self._f: h5py.File | None = None
        try:
            os.unlink(self._part)  # orphan of a cancelled run (see class docstring)
        except OSError:
            # Missing part file, or a missing directory to hold one (the
            # mistyped-single-mode-folder case) — nothing to reclaim either way.
            pass

    def _open(self) -> h5py.File:
        """The part file, created on first use (see the class docstring)."""
        if self._f is None:
            self._f = h5py.File(self._part, "w")
        return self._f

    def _require_open(self, what: str) -> h5py.File:
        if self._f is None:
            raise ValueError(f"cannot {what}: no layer has been appended yet")
        return self._f

    def append(self, dataset_path: str, layer: np.ndarray) -> None:
        """Add one 2-D layer to *dataset_path*, creating the dataset if needed."""
        layer = np.asarray(layer)
        seen = self._shapes.setdefault(dataset_path, set())
        seen.add(tuple(layer.shape))
        f = self._open()
        if dataset_path not in f:
            kw: dict = {}
            if self._compression:
                kw["compression"] = self._compression
                if self._compression == "gzip":
                    kw["compression_opts"] = 4
            f.create_dataset(
                dataset_path,
                shape=(0, *layer.shape),
                maxshape=(None, *layer.shape),
                dtype=layer.dtype,
                chunks=(1, *layer.shape),
                **kw,
            )
        dset = f[dataset_path]
        if tuple(layer.shape) != tuple(dset.shape[1:]):
            raise ValueError(f"{dataset_path}: maps have differing shapes {seen}; fix ROI")
        if layer.dtype != dset.dtype:
            # The dataset's dtype is fixed by the first layer, so a wider layer
            # arriving later would be silently truncated on write (np.stack
            # promoted instead). Refuse rather than corrupt the volume.
            raise ValueError(
                f"{dataset_path}: maps have differing dtypes "
                f"({dset.dtype} for the first layer, {layer.dtype} for this one); "
                "every layer of one volume must share a dtype"
            )
        n = dset.shape[0]
        dset.resize(n + 1, axis=0)
        dset[n] = layer

    def shape(self, dataset_path: str) -> tuple[int, ...]:
        f = self._require_open(f"report the shape of {dataset_path!r}")
        return tuple(int(d) for d in f[dataset_path].shape)

    def datasets(self) -> list[str]:
        return sorted(self._shapes)

    def set_attrs(self, **attrs) -> None:
        f = self._require_open("set file attributes")
        for key, value in attrs.items():
            f.attrs[key] = value

    def close(self) -> None:
        """Flush, close and move the part file into place.

        A no-op once the file has been committed or aborted: a stage that calls
        :meth:`abort` and then ``return``s from inside the ``with`` block leaves
        ``__exit__`` to call ``close()`` on an already-discarded part file, and
        that must not raise. Likewise a no-op when nothing was ever appended —
        there is no part file to commit. A *first* close still propagates
        whatever failed.
        """
        if self._closed:
            return
        if self._f is None:
            self._closed = True
            return
        self._f.close()
        os.replace(self._part, self._path)
        self._closed = True

    def abort(self) -> None:
        """Close and discard; never masks the caller's exception."""
        if self._closed:
            return
        self._closed = True
        if self._f is None:
            return
        try:
            self._f.close()
        except Exception:  # noqa: BLE001 - already-broken file
            pass
        try:
            os.unlink(self._part)
        except OSError:
            pass

    def __enter__(self) -> StackedVolumeFile:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False
