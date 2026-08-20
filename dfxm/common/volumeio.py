"""Bounded-memory volume reading (Qt-free).

One shared implementation so every stage streams the same way instead of eight
divergent schemes. ``mosaicity._streamed_clim`` already streamed layer-by-layer
for plotting; this generalises that pattern and adds a memory budget. (Note that
``mosaicity.run`` itself does NOT stream — it collects four whole volumes.)

The governing guarantee for the **streaming** helpers is **budget-independence**:
for any ``budget_bytes`` they produce bit-identical results. That is what makes a
laptop and a workstation emit the same publishable data product. See
:func:`neumaier_sum` for why ordinary summation would break it.

The **display** helpers (:func:`display_headroom_bytes`,
:func:`display_decimation`, :func:`decimation_note`) are deliberately outside
that guarantee: they coarsen a volume that a render path cannot stream, so a
different budget yields a different stride and therefore different data on
screen. They never touch the stored file.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator

import numpy as np


def volume_bytes(dset) -> int:
    """In-memory size of *dset* if fully loaded, in bytes."""
    n = 1
    for dim in dset.shape:
        n *= int(dim)
    return n * int(dset.dtype.itemsize)


# -- display (render) paths ---------------------------------------------------
# A render path uploads the whole array to VTK, so streaming cannot help it and
# coarsening is the only lever left. These three helpers are the single policy
# both render loaders use — the GUI's 3-D viewer (`gui.viewers._rocking_source`)
# and the rotation-video export child (`dfxm.viewer_jobs._load_volume`) — so the
# on-screen view and the video it exports cannot drift apart in *rule*. They can
# still land on different strides: `display_headroom_bytes()` re-reads available
# RAM, and the export child runs under its own memory pressure. Living here rather
# than in `gui/` is what lets `dfxm/` use them: the GUI depends on the core,
# never the reverse.

# A render loader peaks at roughly TWICE the volume it keeps: the strided read is
# upcast by `.astype(float)` (a transient copy of the same size), and then a
# second full-size array is built alongside it — `vol[np.isfinite(vol)]` for the
# percentile clim in the viewer, `Scene3D.prepared()`'s masked copy for the video.
# Sizing the stride against ONE copy would let a volume landing just under
# headroom take stride 1 and then allocate about twice headroom, which is exactly
# the OOM the decimation exists to prevent.
DISPLAY_COPIES = 2

# At the cap the picture is 16x coarser per axis, i.e. 4096x fewer voxels; not
# reachable in practice (~32 G voxels) but it bounds the loop.
MAX_DISPLAY_DECIMATION = 16


def display_headroom_bytes() -> int:
    """How much RAM a display load (3-D viewer / video export) may use here."""
    from . import advice, machine

    return advice.headroom_bytes(machine.profile())


def display_decimation(dset, budget_bytes: int) -> int:
    """Smallest power-of-two stride bringing *dset* within *budget_bytes* for display.

    ``budget_bytes`` is the machine's headroom; the peak is budgeted at
    :data:`DISPLAY_COPIES` copies of the strided array (see the note above).

    *dset* must be **3-D**: the returned stride is applied as ``[::s, ::s, ::s]``
    by every caller and the budget divides by ``s ** 3``, so the number is only
    meaningful for a volume. A render path has nothing to do with a 2-D dataset
    anyway, and without this guard one reaches the caller's slicing as an
    opaque "too many indices" ``IndexError``.
    """
    ndim = len(dset.shape)
    if ndim != 3:
        raise ValueError(
            f"display decimation needs a 3-D volume, got a {ndim}-D dataset "
            f"with shape {tuple(int(d) for d in dset.shape)}"
        )
    # float64 is what a render path holds, whatever the stored dtype.
    needed = volume_bytes(dset) // max(1, int(dset.dtype.itemsize)) * 8
    budget = max(1, int(budget_bytes) // DISPLAY_COPIES)
    step = 1
    while step < MAX_DISPLAY_DECIMATION and needed // (step**3) > budget:
        step *= 2
    return step


def decimation_note(step: int, full_shape) -> str:
    """The user-facing sentence for a display load that was decimated.

    ``full_shape`` is printed in the dataset's own ``(z, y, x)`` order — the same
    order the viewer's status line prints the loaded shape in, so the two read as
    one statement instead of contradicting each other.
    """
    shape = tuple(int(d) for d in full_shape)
    return (
        f"decimated {step}x for display (full shape {shape} exceeds this machine's "
        "memory headroom) — the stored data is unchanged"
    )


def _layers_per_block(dset, budget_bytes: int, axis: int) -> int:
    """How many slices along *axis* fit in the budget. Always at least 1."""
    n_layers = int(dset.shape[axis])
    if n_layers <= 0:
        return 1
    per_layer = max(1, volume_bytes(dset) // n_layers)
    return max(1, min(n_layers, int(max(1, budget_bytes) // per_layer)))


def iter_blocks(dset, *, budget_bytes: int, axis: int = 0) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield ``(slice, array)`` blocks along *axis*, each within the budget.

    Blocks are yielded in ascending order and together cover the dataset
    exactly once, so concatenating them along *axis* reproduces the whole
    dataset. A budget smaller than one slice still yields single slices rather
    than stalling — progress always beats precision here.
    """
    if axis not in (0, 1):
        raise ValueError(f"only axis=0 and axis=1 blocking are supported, got {axis}")
    n_layers = int(dset.shape[axis])
    step = _layers_per_block(dset, budget_bytes, axis)
    for start in range(0, n_layers, step):
        sl = slice(start, min(start + step, n_layers))
        yield sl, (dset[sl] if axis == 0 else dset[:, sl])


def iter_with_context(blocks, *, trailing: int = 1, axis: int = 0):
    """Re-yield *blocks*, each carrying the next block's first rows.

    Yields ``(interior, window, within)``: ``window`` is the block with
    *trailing* rows of its successor appended, ``interior`` is the block's own
    range in source coordinates, and ``within`` indexes that range inside
    ``window``, so a consumer writes ``window[within]`` and never redoes the
    arithmetic. The final block gets no context, which is correct — it ends
    where the source ends, so its edge behaviour must match the source's.

    This is what lets an operation with a forward-looking local dependency
    stream: linear interpolation and ``map_coordinates(order=1)`` both read
    the row after the one they land on.

    It takes a *stream* of blocks rather than a dataset deliberately. The
    blocks needing context are often generated — an aligned volume that is
    never materialised — and cannot be re-indexed to widen a read window.

    *axis* must match the axis the stream was blocked along (see
    :func:`iter_blocks`), because "the next rows" means the next rows *along
    that axis*. Getting it wrong is not a shape error that stops you: over an
    ``axis=1`` stream an axis-0 implementation still recovers each interior
    correctly — ``within`` happens to span the whole of axis 0 — and only the
    appended context is wrong, silently supplying the successor's first
    Z-layer where its first column was meant. Hence the parameter rather than
    an assumption. ``within`` is a plain slice for ``axis=0`` and a tuple for
    ``axis=1``; either way ``window[within]`` is the interior.

    Raises ``ValueError`` if a successor cannot supply *trailing* rows. The
    alternative — handing back a window one row short — corrupts exactly one
    block edge and hides from the budget-independence tests, since every
    budget corrupts it identically.

    Memory cost is *trailing* rows above the block itself, so a budget is
    exceeded by that much; with the default of one row against any realistic
    block that is negligible, but it is stated because an unstated
    approximation in a memory-budget module is how budgets stop being trusted.
    """
    if axis not in (0, 1):
        raise ValueError(f"only axis=0 and axis=1 context are supported, got {axis}")

    def _interior(block):
        n = block.shape[axis]
        return slice(0, n) if axis == 0 else (slice(None), slice(0, n))

    previous = None
    for sl, block in blocks:
        if previous is not None:
            prev_sl, prev_block = previous
            head = block[:trailing] if axis == 0 else block[:, :trailing]
            available = head.shape[axis]
            if available < trailing:
                raise ValueError(
                    f"block {sl} cannot supply the trailing={trailing} rows of context "
                    f"requested along axis={axis}: it has only {available}. Use a larger "
                    "budget so blocks exceed the context width, or a smaller trailing."
                )
            window = np.concatenate([prev_block, head], axis=axis) if trailing else prev_block
            yield prev_sl, window, _interior(prev_block)
        previous = (sl, block)
    if previous is not None:
        prev_sl, prev_block = previous
        yield prev_sl, prev_block, _interior(prev_block)


class BlockReader:
    """A dataset too large for the budget, presented as a stream of blocks.

    Deliberately *not* an ndarray look-alike: code receiving one must handle
    blocks explicitly, so an accidental whole-volume materialisation is a
    visible change rather than a silent one.
    """

    def __init__(self, dset, budget_bytes: int, axis: int = 0) -> None:
        self._dset = dset
        self._budget = budget_bytes
        self._axis = axis
        self.shape = tuple(int(d) for d in dset.shape)
        self.dtype = dset.dtype

    def __iter__(self) -> Iterator[tuple[slice, np.ndarray]]:
        return iter_blocks(self._dset, budget_bytes=self._budget, axis=self._axis)

    @property
    def nbytes(self) -> int:
        return volume_bytes(self._dset)


def load_or_stream(dset, *, budget_bytes: int):
    """The whole array when it fits the budget, else a :class:`BlockReader`."""
    if volume_bytes(dset) <= budget_bytes:
        return dset[:]
    return BlockReader(dset, budget_bytes)


def neumaier_sum(values, *, state: tuple[float, float] | None = None) -> tuple[float, float]:
    """Compensated sum of *values*, continuable across blocks.

    Returns ``(total, compensation)``; the true sum is ``total + compensation``.
    Pass a previous return value back as *state* to continue an accumulation —
    that is what makes the result independent of how the data was blocked.

    Why not ``np.sum``: numpy reduces pairwise with a 128-element base case, so
    summing an array whole and summing it in blocks give different bits. This
    walks elements in a fixed order with an explicit compensation term, so the
    answer depends only on the data — never on the memory budget.
    """
    total, comp = state if state is not None else (0.0, 0.0)
    for value in np.asarray(values, dtype=np.float64).ravel():
        item = float(value)
        tentative = total + item
        if abs(total) >= abs(item):
            comp += (total - tentative) + item
        else:
            comp += (item - tentative) + total
        total = tentative
    return total, comp


def dataset_blocks(dset, *, budget_bytes: int, axis: int = 0) -> Iterator[np.ndarray]:
    """Just the arrays from :func:`iter_blocks`, for feeding the reductions."""
    for _sl, block in iter_blocks(dset, budget_bytes=budget_bytes, axis=axis):
        yield block


def stream_mean(blocks) -> float:
    """Mean of the finite values across *blocks*, bit-identical for any blocking.

    Takes an iterable of arrays rather than a dataset, because the statistics
    this serves are over *generated* blocks — the aligned volume a stage never
    materialises — not over anything on disk.
    """
    state = (0.0, 0.0)
    count = 0
    for block in blocks:
        finite = block[np.isfinite(block)]
        if finite.size:
            state = neumaier_sum(finite, state=state)
            count += int(finite.size)
    if not count:
        return float("nan")
    return (state[0] + state[1]) / count


def stream_minmax(blocks) -> tuple[float, float]:
    """Min and max of the finite values across *blocks*."""
    lo, hi = np.inf, -np.inf
    for block in blocks:
        finite = block[np.isfinite(block)]
        if finite.size:
            lo = min(lo, float(finite.min()))
            hi = max(hi, float(finite.max()))
    if not np.isfinite(lo):
        return float("nan"), float("nan")
    return lo, hi


def block_reduce(dset, fn, *, budget_bytes: int, init):
    """Fold *dset* block-by-block with ``fn(acc, block) -> acc``.

    Budget-independence requires *fn* to be **partition-invariant**, which is
    stricter than intuitive associativity: ``fn(fn(acc, A), B)`` must
    *bit-equal* ``fn(acc, concat(A, B))`` for adjacent blocks. Per-block
    ``np.sum`` fails exactly this (pairwise ordering changes with the block
    size) even though summation is "associative" on paper — carry compensated
    state via :func:`neumaier_sum` instead. Blocks always arrive in ascending
    order.
    """
    acc = init
    for _sl, block in iter_blocks(dset, budget_bytes=budget_bytes):
        acc = fn(acc, block)
    return acc


def block_nansum(dset, *, budget_bytes: int) -> float:
    """Sum of the FINITE values of *dset*, bit-identical for any budget.

    Ignores all non-finite values — NaN *and* ±inf — unlike ``np.nansum``,
    which propagates inf. (Propagating inf through compensated summation would
    poison the compensation term to NaN; dropping non-finite values is the
    right semantics for the DFXM statistics this serves.) On all-finite data
    it differs from ``np.nansum`` by up to ~1 ulp — deliberately: budget-
    independence is the property worth having; matching numpy's pairwise
    ordering is not.
    """

    def fold(acc, block):
        finite = block[np.isfinite(block)]
        return neumaier_sum(finite, state=acc)

    total, comp = block_reduce(dset, fold, budget_bytes=budget_bytes, init=(0.0, 0.0))
    return total + comp


def two_pass(dset, stat_fn, apply_fn, *, budget_bytes: int, init):
    """Global statistic first, then apply it block-wise.

    Pass 1 folds every block through ``stat_fn(acc, block) -> acc``. Pass 2
    yields ``(slice, apply_fn(stat, block))`` for each block. Costs one extra
    read of the dataset — the price of a lossless global operation in bounded
    memory.
    """
    stat = block_reduce(dset, stat_fn, budget_bytes=budget_bytes, init=init)
    for sl, block in iter_blocks(dset, budget_bytes=budget_bytes):
        yield sl, apply_fn(stat, block)


@contextlib.contextmanager
def scratch_array(shape, dtype, *, dirpath: str, prefix: str = "dfxm_scratch"):
    """A disk-backed working array, deleted on exit even if the caller raises.

    Yields a ``np.memmap`` — an ``ndarray`` subclass, so numerical code needs no
    changes; its pages spill to disk instead of occupying RAM. This is what lets
    irreducibly whole-array work (an exact median, a global fit) run on a
    machine that cannot hold the volume: slower, but it finishes.

    Caveat the caller must respect: **temporaries still allocate in RAM**.
    ``out = a * b + c`` on a memmapped ``a`` materialises a full-size temporary
    and defeats the purpose. Bucket-3 code must operate in slabs with in-place
    operations (``np.multiply(a, b, out=a)``).

    Second caveat: **the array is dead once the block exits.** Never return it
    or stash a reference for later — the backing file is deleted on exit, and
    on Windows the mapping is explicitly closed first, so a post-exit element
    access there is a hard process-killing fault (on POSIX the pages merely
    outlive the unlinked file until garbage collection).
    """
    os.makedirs(dirpath, exist_ok=True)
    handle, path = tempfile.mkstemp(prefix=prefix, suffix=".dat", dir=dirpath)
    os.close(handle)
    memmap = None
    try:
        memmap = np.memmap(path, dtype=dtype, mode="w+", shape=tuple(shape))
        yield memmap
    finally:
        # Windows refuses to unlink a file while it is still mapped, so there —
        # and only there — the mapping is force-closed before deleting, which
        # turns any lingering caller reference into a hard fault on access.
        # POSIX unlinks mapped files fine, so the close is skipped and a stale
        # reference reads harmless (soon-to-be-collected) pages instead of
        # segfaulting.
        if memmap is not None:
            try:
                memmap.flush()
            except Exception:  # noqa: BLE001 - already-broken mapping
                pass
            if os.name == "nt" and getattr(memmap, "_mmap", None) is not None:
                try:
                    memmap._mmap.close()
                except Exception:  # noqa: BLE001
                    pass
            del memmap
            import gc

            gc.collect()
        try:
            os.unlink(path)
        except OSError:
            pass  # never mask the caller's exception with a cleanup failure
