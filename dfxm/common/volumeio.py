"""Bounded-memory volume reading (Qt-free).

One shared implementation so every stage streams the same way instead of eight
divergent schemes. ``mosaicity._volume_stats`` already streamed layer-by-layer
for plotting; this generalises that pattern and adds a memory budget. (Note that
``mosaicity.run`` itself does NOT stream — it collects four whole volumes.)

The governing guarantee is **budget-independence**: for any ``budget_bytes``,
these helpers produce bit-identical results. That is what makes a laptop and a
workstation emit the same publishable data product. See :func:`block_reduce`
for why ordinary summation would break it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def volume_bytes(dset) -> int:
    """In-memory size of *dset* if fully loaded, in bytes."""
    n = 1
    for dim in dset.shape:
        n *= int(dim)
    return n * int(dset.dtype.itemsize)


def _layers_per_block(dset, budget_bytes: int, axis: int) -> int:
    """How many slices along *axis* fit in the budget. Always at least 1."""
    n_layers = int(dset.shape[axis])
    if n_layers <= 0:
        return 1
    per_layer = max(1, volume_bytes(dset) // n_layers)
    return max(1, min(n_layers, int(max(1, budget_bytes) // per_layer)))


def iter_blocks(dset, *, budget_bytes: int, axis: int = 0) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield ``(slice, array)`` blocks along *axis*, each within the budget.

    Blocks are yielded in ascending order and together cover the dataset exactly
    once, so concatenating them along *axis* reproduces the whole volume. A
    budget smaller than one layer still yields single layers rather than
    stalling — progress always beats precision here.
    """
    if axis != 0:
        raise ValueError("only axis=0 blocking is supported")
    n_layers = int(dset.shape[0])
    step = _layers_per_block(dset, budget_bytes, axis)
    for start in range(0, n_layers, step):
        stop = min(start + step, n_layers)
        sl = slice(start, stop)
        yield sl, dset[sl]


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


def block_reduce(dset, fn, *, budget_bytes: int, init):
    """Fold *dset* block-by-block with ``fn(acc, block) -> acc``.

    *fn* must be associative in the order blocks are produced (they always
    arrive in ascending order), so the result is budget-independent.
    """
    acc = init
    for _sl, block in iter_blocks(dset, budget_bytes=budget_bytes):
        acc = fn(acc, block)
    return acc


def block_nansum(dset, *, budget_bytes: int) -> float:
    """NaN-ignoring sum of *dset*, bit-identical for any budget.

    Differs from ``np.nansum`` by up to ~1 ulp — deliberately. Budget-
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
