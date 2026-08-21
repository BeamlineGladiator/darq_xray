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


# How many independent compensated accumulators :func:`neumaier_sum` keeps.
# The value at global position *i* of the stream always lands in lane
# ``i % NEUMAIER_LANES`` — see :class:`NeumaierState` for why that "global", and
# not "position within this block", is the whole guarantee.
#
# 4096 is a measured optimum, not a round number, and the curve has a floor in
# the middle rather than running one way: fewer lanes means more Python
# iterations (the cost this replaced), while more lanes push the six length-N
# temporaries of a fold out of L2 cache. Summing 15 M float64, best of three:
#
#   lanes:    256     1024     4096    16384    65536   262144
#   time:  0.492 s  0.179 s  0.093 s  0.333 s  0.346 s  0.377 s
#
# against 2.19 s for the per-element loop this replaced and 0.115 s for
# `np.nansum` — i.e. the compensated sum is no longer the expensive option. The
# *answer* was bit-identical at every lane count in that sweep, so the constant
# is a performance knob and not part of the result.
NEUMAIER_LANES = 1 << 12


def _fold_lanes(total: np.ndarray, comp: np.ndarray, values: np.ndarray, lane: int) -> None:
    """One Neumaier step for lanes ``[lane, lane + values.size)``, in place.

    Elementwise, this is exactly the scalar recurrence — ``s = t + v``, then
    ``c += (larger - s) + smaller`` — so a lane's arithmetic does not depend on
    how many lanes ran beside it. That is what makes the vector form bit-equal
    to running each lane on its own, which is in turn what lets the lane count
    be an implementation detail rather than part of the answer.

    The magnitude branch is expressed as a *selection* of operands rather than
    as two arithmetic arms fed to `np.where`, so there is one arithmetic path
    to compare against the scalar recurrence instead of two, only one of which
    survives. (It does not make the function total on infinite input: the
    surviving path still evaluates ``inf - inf`` there, exactly as the scalar
    recurrence does — see the `np.errstate` in :func:`neumaier_sum`.)
    """
    stop = lane + values.size
    running = total[lane:stop]
    tentative = running + values
    bigger_first = np.abs(running) >= np.abs(values)
    larger = np.where(bigger_first, running, values)
    smaller = np.where(bigger_first, values, running)
    comp[lane:stop] += (larger - tentative) + smaller
    total[lane:stop] = tentative


class NeumaierState:
    """The continuable state of a compensated sum: one accumulator per lane.

    Not a plain ``(total, compensation)`` pair, because a pair cannot be
    continued without collapsing the lanes — and collapsing them at a block
    boundary would make the answer depend on where that boundary fell, which is
    precisely what this module refuses to allow. Indexing (``state[0]``,
    ``state[1]``) and unpacking still yield the reduced total and compensation,
    whose sum is :attr:`value`.

    Two states compare equal when their lanes are equal **bit for bit** and they
    have consumed the same number of values. That is deliberately stricter than
    comparing the sums: it is the assertion that catches a lane scheme which
    regrouped the data and happened to get away with it in the total.
    """

    __slots__ = ("_comp", "_count", "_total")

    def __init__(self, total=None, comp=None, count: int = 0) -> None:
        self._total = np.zeros(NEUMAIER_LANES) if total is None else total
        self._comp = np.zeros(NEUMAIER_LANES) if comp is None else comp
        self._count = int(count)

    @property
    def count(self) -> int:
        """How many values have been folded in — the stream's global position."""
        return self._count

    def copy(self) -> NeumaierState:
        return NeumaierState(self._total.copy(), self._comp.copy(), self._count)

    def _reduce(self) -> tuple[float, float]:
        """Fold the lanes into one ``(total, compensation)`` pair.

        A per-element Neumaier again, but over ``2 * NEUMAIER_LANES`` numbers
        rather than over the volume — ~1 ms, once, at the end of a reduction
        instead of ~155 ns per voxel throughout it. Lanes are walked in lane
        order, so the reduction is as blocking-blind as the accumulation.
        """
        total = comp = 0.0
        for item in np.concatenate([self._total, self._comp]).tolist():
            tentative = total + item
            if abs(total) >= abs(item):
                comp += (total - tentative) + item
            else:
                comp += (item - tentative) + total
            total = tentative
        return total, comp

    @property
    def value(self) -> float:
        """The sum itself."""
        total, comp = self._reduce()
        return total + comp

    def __getitem__(self, index: int) -> float:
        return self._reduce()[index]

    def __iter__(self):
        return iter(self._reduce())

    def __len__(self) -> int:
        return 2

    def __eq__(self, other) -> bool:
        if not isinstance(other, NeumaierState):
            return NotImplemented
        return (
            self._count == other._count
            # `tobytes`, not `array_equal`: `-0.0 == 0.0` there, and a lane whose
            # sign flipped with the blocking is exactly the kind of drift this
            # comparison exists to expose (see `stream_minmax`).
            and self._total.tobytes() == other._total.tobytes()
            and self._comp.tobytes() == other._comp.tobytes()
        )

    def __repr__(self) -> str:
        return f"NeumaierState(value={self.value!r}, count={self._count})"


def neumaier_sum(values, *, state: NeumaierState | None = None) -> NeumaierState:
    """Compensated sum of *values*, continuable across blocks.

    Returns a :class:`NeumaierState`; the sum is its ``.value`` (equivalently
    ``state[0] + state[1]``). Pass a previous return value back as *state* to
    continue an accumulation — that is what makes the result independent of how
    the data was blocked. The state is never mutated; a fresh one comes back.

    Why not ``np.sum``: numpy reduces pairwise with a 128-element base case, so
    summing an array whole and summing it in blocks give different bits. This
    keeps :data:`NEUMAIER_LANES` compensated accumulators and sends the value at
    global position *i* to lane ``i % NEUMAIER_LANES``, whatever block it
    arrived in — so the answer depends only on the data and its position in the
    stream, never on the memory budget.

    The lane bookkeeping is the load-bearing part. An incoming run is split
    into three: the values completing the lane row the previous call stopped
    part-way through, the whole rows after it, and the ragged remainder. Each
    piece is folded into the lane range its *global* indices name. Reshaping a
    run onto lane 0 instead would be simpler, would pass every whole-array test,
    and would regroup the data the moment a block boundary stopped landing on a
    lane boundary.

    Accuracy is the same class as the per-element order it replaced, and on
    realistic data the same *bits*: over hundreds of random samples and every
    DFXM-shaped volume tried, the lane order and the per-element order agree
    exactly, and both may differ from ``np.nansum``'s pairwise order by a few
    ulps. They are not identical in general — constructed catastrophic
    cancellation (`a`, `-a`, small, at `a ~ 1e16`) separates them, and there
    neither one is the correctly-rounded sum, because Neumaier's single
    compensation term is not exact in that regime. What holds unconditionally is
    that the answer never moves with the budget.

    Memory cost, all of it constant in the size of the data: the lane state is
    ``2 * NEUMAIER_LANES`` float64 (64 kB), copied once per call, and a fold
    allocates about six further arrays of at most ``NEUMAIER_LANES`` float64
    (~200 kB, transient). Roughly a quarter of a megabyte in total. Stated
    because an unstated allocation in the memory-budget module is how budgets
    stop being trusted; against a block of even one volume layer it is noise.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    result = NeumaierState() if state is None else state.copy()
    if flat.size == 0:
        return result
    total, comp = result._total, result._comp
    lane = result._count % NEUMAIER_LANES
    taken = 0
    # Python float arithmetic reports neither of the two IEEE conditions this
    # recurrence can reach; numpy reports both, and under `-W error` a report
    # becomes an exception where the per-element loop returned a number. Keeping
    # this a pure speed change therefore means silencing exactly those two:
    #
    #   invalid — `inf - inf`, on an infinite input, which poisons the
    #     compensation to NaN in the scalar recurrence too;
    #   over    — `t + v` leaving float64 range, which the scalar recurrence
    #     also does, silently, on its way to ±inf.
    #
    # Neither is reachable from this module's own callers, which filter
    # non-finite values out first, and `over` needs finite values whose partial
    # sums exceed ~1.8e308.
    #
    # On overflowing input the vector and scalar orders can disagree in VALUE,
    # not merely in reporting: on `[1e308, 1, -1e308, 1] * 5000` the scalar order
    # cancels each pair immediately and returns 10000.0, while here the period 4
    # divides NEUMAIER_LANES, so one lane meets 1e308 over and over, overflows to
    # inf and yields NaN. That divergence is a function of the data, not of the
    # budget — every budget gives the same NaN — so budget-independence holds. It
    # is documented rather than engineered away: no DFXM quantity comes near
    # 1e308, and defending against it would cost the whole speedup.
    with np.errstate(invalid="ignore", over="ignore"):
        if lane:  # finish the row the last call left part-way through
            taken = min(flat.size, NEUMAIER_LANES - lane)
            _fold_lanes(total, comp, flat[:taken], lane)
        rows = (flat.size - taken) // NEUMAIER_LANES
        if rows:
            matrix = flat[taken : taken + rows * NEUMAIER_LANES].reshape(rows, NEUMAIER_LANES)
            for row in matrix:
                _fold_lanes(total, comp, row, 0)
            taken += rows * NEUMAIER_LANES
        if taken < flat.size:  # the ragged tail, into the lanes it globally belongs to
            _fold_lanes(total, comp, flat[taken:], 0)
    result._count += int(flat.size)
    return result


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
    state = None
    for block in blocks:
        finite = block[np.isfinite(block)]
        if finite.size:
            state = neumaier_sum(finite, state=state)
    if state is None or state.count == 0:
        return float("nan")
    # `.value` once, not `state[0] + state[1]`: each read of the pair folds the
    # lanes again, and the fold is the one per-element loop left in here.
    return state.value / state.count


def stream_minmax(blocks) -> tuple[float, float]:
    """Min and max of the finite values across *blocks*.

    The sign of a zero bound is decided by the data, never by the blocking:
    the minimum prefers ``-0.0`` and the maximum ``+0.0`` whenever that sign is
    present among the zeros. Without this the sign is a *budget-dependent* bit,
    because ``-0.0 == 0.0`` means each block-wise `min`/`max` simply keeps
    whichever it met first — on ``[-0.0]*8 + [0.0]*8 + [1, 2]`` this returned
    ``+0.0`` over the whole array and ``-0.0`` at every smaller budget. There
    is no numpy parity at stake (``np.min`` over mixed zeros is equally
    arbitrary) and no numerical information either way, since the two compare
    equal; what is at stake is the module's governing guarantee, and this is
    the one violation of it the project's own harness cannot see —
    `tests/equivalence.py` notes that `array_equal` calls the two zeros equal.

    The extra scan runs only for a block whose own min or max is exactly zero.
    """
    lo, hi = np.inf, -np.inf
    negative_zero_at_lo = False  # a `-0.0` sits at the minimum
    positive_zero_at_hi = False  # a `+0.0` sits at the maximum
    for block in blocks:
        finite = block[np.isfinite(block)]
        if not finite.size:
            continue
        block_lo, block_hi = float(finite.min()), float(finite.max())
        if block_lo == 0.0 or block_hi == 0.0:
            # `np.signbit(finite)`, not `np.signbit(finite[finite == 0.0])`:
            # selecting the zeros first copies them, which on a zero-heavy
            # block is a second full-size array — 16.8 MB extra on a 16.8 MB
            # block, inside the one module whose job is to respect a memory
            # budget, on exactly the data it was hardened for (the paraview
            # volume's masked voxels are exactly 0.0, so this branch fires in
            # every block). Looking at every value costs one byte each and is
            # equivalent here: a block whose min is zero holds no negative
            # value, so a set sign bit can only be a `-0.0`; a block whose max
            # is zero holds no positive value, so a clear one can only be a
            # `+0.0`. Those are also the blocks that can hold the global
            # extreme, which is what makes the flags correct.
            signs = np.signbit(finite)
            negative_zero_at_lo |= block_lo == 0.0 and bool(signs.any())
            positive_zero_at_hi |= block_hi == 0.0 and not bool(signs.all())
        lo = min(lo, block_lo)
        hi = max(hi, block_hi)
    if not np.isfinite(lo):
        return float("nan"), float("nan")
    if lo == 0.0:
        lo = -0.0 if negative_zero_at_lo else 0.0
    if hi == 0.0:
        hi = 0.0 if positive_zero_at_hi else -0.0
    return lo, hi


_QUANTILE_BINS = 1 << 16
_QUANTILE_EXACT_CAP = 1 << 20


def _finite64(block) -> np.ndarray:
    """The finite values of *block*, as float64.

    The cast is load-bearing, not tidiness. numpy 1.x compares a float32 array
    against a Python float by **value-based casting** — it demotes the scalar
    to float32 — so a float64 bin edge silently rounds and a mask can exclude
    a value that the float64 `searchsorted` counted *into* that bin. The
    counted bin and the collected bin then disagree: an empty survivor set (a
    bare `ValueError` out of `np.concatenate`) at best, a selection from the
    wrong candidates at worst. This project stores volumes as float32, so that
    is the common case, not the exotic one. Widening float32 to float64 is
    exact — it moves no order statistic — and it makes every comparison in
    this file happen in one dtype. The cost is one float64 copy of a block's
    finite values (2× the block for float32), which the mask made anyway.
    """
    return np.asarray(block[np.isfinite(block)], dtype=np.float64)


def _zero_sign(make_blocks) -> float:
    """``-0.0`` when every zero in the data is negative, otherwise ``+0.0``.

    ``-0.0`` and ``0.0`` compare equal, so every `min`, `sort` and dict key on
    the way to an answer keeps whichever it *met first* — which the budget
    decides, since it decides the blocking. That is a budget-dependent bit in
    the result, invisible to `assert_budget_independent` (`array_equal` calls
    the two equal) but plain in `float.hex()`.

    Settling it from the data restores both properties at once. When every
    zero present is negative there is no ambiguity and ``-0.0`` is also what
    numpy reports, so the bit-for-bit match holds. When both signs are present
    numpy's own answer comes from an arbitrary partition order — there is no
    bit to match without sorting the whole volume — so ``+0.0`` is chosen and
    the answer at least stops moving with the budget.

    Costs one traversal, and only when the answer is a zero.
    """
    for block in make_blocks():
        finite = _finite64(block)
        zeros = finite[finite == 0.0]
        if zeros.size and not np.signbit(zeros).all():
            return 0.0
    return -0.0


def _lerp(lo: float, hi: float, t: float, dtype=np.float64) -> float:
    """numpy's linear interpolation for percentiles, reproduced bit-for-bit.

    Two details make this the same number rather than a very close one.
    numpy switches formula at ``t >= 0.5`` for numerical stability; and it
    computes ``hi - lo`` **in the array's own dtype** before widening, so on
    float32 data — what this project stores — that difference is rounded to
    float32 first. Subtracting in float64 instead diverges by ~1e-8 relative
    whenever the two bracketing order statistics differ by more than about 2×,
    which a heavy-tailed, heavily-masked volume produces readily.

    *dtype* is the data's dtype. Non-floating dtypes fall back to float64:
    numpy would subtract in the integer type, which agrees with float64 for
    every value below 2**53 and differs only by overflowing.
    """
    work = np.dtype(dtype) if np.issubdtype(dtype, np.floating) else np.dtype(np.float64)
    lo_w, hi_w = np.asarray(lo, dtype=work), np.asarray(hi, dtype=work)
    diff = np.subtract(hi_w, lo_w)  # in the data's dtype, as numpy does
    result = np.add(np.asarray(lo, dtype=np.float64), diff * np.float64(t))
    if t >= 0.5:
        # No `hi != lo` guard: numpy has none, and the two forms differ for
        # equal endpoints in exactly one case — the sign of a zero, which form
        # 1 normalises to `+0.0` and form 2 leaves as `-0.0`.
        result = np.subtract(np.asarray(hi, dtype=np.float64), diff * np.float64(1.0 - t))
    return float(result)


def stream_quantile(make_blocks, q: float) -> float:
    """The *q*-th percentile of the finite values, exactly, in bounded memory.

    Returns what ``np.percentile(finite_values, q)`` returns — not an estimate.
    Colour limits are computed this way, so an approximation would shift every
    existing figure's colours; exactness is the requirement.

    Pass 1 takes the finite min, max and count. Pass 2 histograms into that
    range and locates the bin holding the target rank. Pass 3 collects the
    survivors of that bin — a small array — and selects exactly. A bin holding
    more than ``_QUANTILE_EXACT_CAP`` values is narrowed into and the histogram
    repeated, at two traversals a round (see :func:`_observed_bounds`); with
    65536 bins that is uncommon, and a handful of rounds when it happens.

    *make_blocks* is a zero-argument callable returning a fresh iterable of
    arrays, because the algorithm traverses more than once.

    Working set is a few tens of MB above the caller's block, whatever the
    volume's size: the histogram's 65537 edges and counts, the survivors of
    one bin (at most ``_QUANTILE_EXACT_CAP`` values) held twice while they are
    concatenated and sorted, and — for float32 input — the float64 widening of
    a block's finite values (see :func:`_finite64`).

    One deliberate divergence from ``np.percentile``: on a rank that needs no
    interpolation the order statistic is returned as-is, so a spread wide
    enough to overflow ``hi - lo`` gives that value rather than the ``nan``
    numpy's unconditional lerp produces there. Every rank that does
    interpolate reproduces numpy, overflow included.

    The sign of a zero answer follows :func:`_zero_sign`, not a blanket
    normalisation: ``-0.0`` when every zero in the data is negative — which is
    what numpy returns for such an array, and it is *not* always ``+0.0``,
    since numpy's form 2 (taken at ``t >= 0.5``) keeps the sign — and ``+0.0``
    when both signs are present, where numpy's own answer comes from an
    arbitrary partition order and there is no bit to match. Normalising
    unconditionally would look tidier and would break numpy parity.
    """
    if not 0.0 <= q <= 100.0:
        # numpy raises here; returning a clamped answer instead would be a
        # silently wrong number, the one failure mode this helper must not have.
        raise ValueError(f"Percentiles must be in the range [0, 100], got {q!r}")
    lo, hi = stream_minmax(make_blocks())
    if not np.isfinite(lo):
        return float("nan")
    count = 0
    dtype = None
    for block in make_blocks():
        array = np.asarray(block)
        dtype = array.dtype if dtype is None else np.promote_types(dtype, array.dtype)
        count += int(np.count_nonzero(np.isfinite(array)))
    # numpy's rank convention for the "linear" method. The expression is
    # numpy's own `get_virtual_index` — `(n - 1) * (q / 100)`, evaluated in
    # that grouping. The algebraically equal `n*qq + (1 - qq) - 1` (numpy's
    # generic Hyndman-Fan form, which "linear" does *not* take) differs in the
    # last bits, and the last bits are exactly what this function promises.
    pos = (q / 100.0) * (count - 1)
    rank_lo = int(np.floor(pos))
    frac = pos - rank_lo

    if count == 1 or lo == hi:
        # Every value is identical, so both bracketing ranks are `lo` and no
        # selection is needed. Still go through `_lerp`: with equal endpoints
        # it is a no-op for every value except a zero, whose sign numpy decides
        # by which of the two forms the fraction selects.
        if lo == 0.0:
            lo = _zero_sign(make_blocks)
        return _lerp(lo, lo, frac, dtype)

    lo_val = _select_rank(make_blocks, rank_lo, lo, hi)
    if frac == 0.0:
        # The rank is exact, so there is nothing to interpolate — and returning
        # here also skips a whole second selection (several more traversals of
        # the volume) for a neighbour that would carry zero weight. q=0 and
        # q=100 always land here. `+ 0.0` is exact and gives a zero the sign
        # numpy's zero-weight lerp would have given it (form 1 normalises).
        return float(lo_val) + 0.0
    # A non-zero fraction means `pos` is not an integer, so `rank_lo` is at most
    # `count - 2` and its successor is always in range.
    hi_val = _select_rank(make_blocks, rank_lo + 1, lo, hi)
    if lo_val == 0.0 or hi_val == 0.0:
        # A selected zero carries whichever sign the selection path met first,
        # which the blocking decides; settle it from the data before the sign
        # reaches `_lerp`, which propagates it (see :func:`_zero_sign`).
        zero = _zero_sign(make_blocks)
        lo_val = zero if lo_val == 0.0 else lo_val
        hi_val = zero if hi_val == 0.0 else hi_val
    return _lerp(lo_val, hi_val, frac, dtype)


def _bin_edges(lo: float, hi: float) -> np.ndarray:
    """`_QUANTILE_BINS` + 1 sorted edges spanning ``[lo, hi]`` inclusive.

    A blend (``lo * (1 - t) + hi * t``) rather than ``np.linspace``, which
    computes ``hi - lo`` first and therefore overflows to inf — and then to nan
    edges, an all-empty histogram and a bare `ValueError` out of
    `np.concatenate` — on a range wider than float64 can subtract. The blend
    stays inside ``[lo, hi]`` for every weight. Rounding can still leave two
    neighbours out of order by an ulp, which `np.searchsorted` would silently
    mis-answer, so the edges are clamped and made non-decreasing.
    """
    weights = np.linspace(0.0, 1.0, _QUANTILE_BINS + 1)
    edges = np.clip(lo * (1.0 - weights) + hi * weights, lo, hi)
    edges = np.maximum.accumulate(edges)
    edges[0], edges[-1] = lo, hi
    return edges


def _select_rank(make_blocks, rank: int, lo: float, hi: float) -> float:
    """The *rank*-th smallest finite value (0-based), by histogram refinement."""
    below = 0  # count of finite values strictly below `lo`
    while True:
        if lo == hi:
            # A fast path, not the termination guarantee: a window that has
            # collapsed entirely holds one value, so return it without
            # histogramming a zero-width range. What actually guarantees
            # termination is the unsplittable-window branch below, which also
            # covers this case (every edge equal, so the chosen bin is the
            # whole window) should this ever be removed.
            return float(lo)
        edges = _bin_edges(lo, hi)
        counts = np.zeros(_QUANTILE_BINS, dtype=np.int64)
        for block in make_blocks():
            finite = _finite64(block)
            window = finite[(finite >= lo) & (finite <= hi)]
            if window.size:
                idx = np.clip(
                    np.searchsorted(edges, window, side="right") - 1, 0, _QUANTILE_BINS - 1
                )
                counts += np.bincount(idx, minlength=_QUANTILE_BINS)
        cumulative = np.cumsum(counts)
        target = rank - below
        bin_index = int(np.searchsorted(cumulative, target, side="right"))
        bin_index = min(bin_index, _QUANTILE_BINS - 1)
        in_bin = int(counts[bin_index])
        before = int(cumulative[bin_index - 1]) if bin_index else 0
        bin_lo, bin_hi = float(edges[bin_index]), float(edges[bin_index + 1])
        last = bin_index == _QUANTILE_BINS - 1
        if in_bin <= _QUANTILE_EXACT_CAP:
            survivors = []
            for block in make_blocks():
                finite = _finite64(block)
                mask = (finite >= bin_lo) & ((finite <= bin_hi) if last else (finite < bin_hi))
                if mask.any():
                    survivors.append(finite[mask])
            values = np.sort(np.concatenate(survivors))
            return float(values[target - before])
        # Too many to sort: narrow and go again — to the bin's *observed*
        # [min, max] rather than to its edges. Edges alone narrow the range
        # geometrically but the *exponent* only linearly, so converging onto a
        # tie at 0.0 walks the whole denormal range: a masked background of
        # exact zeros cost ~144 traversals of the volume, versus 16 for a
        # background of 1.0. On the 17 GB paraview volume, whose masked voxels
        # are exactly 0.0, that is hours of reading — indistinguishable from a
        # hang. Observed bounds collapse a tied bin to `lo == hi` in one round,
        # for one extra traversal per narrowing (and narrowing is rare).
        new_lo, new_hi = _observed_bounds(make_blocks, bin_lo, bin_hi, last)
        if not np.isfinite(new_lo):  # counted but nothing collected: keep the edges
            new_lo, new_hi = bin_lo, bin_hi
        if new_lo == lo and new_hi == hi:
            return _select_among_ties(make_blocks, target, lo, hi)
        below += before
        lo, hi = new_lo, new_hi


def _observed_bounds(make_blocks, bin_lo: float, bin_hi: float, last: bool):
    """The smallest and largest values actually present in one histogram bin.

    Uses the same membership rule as the survivor pass — half-open, closed on
    the last bin — so the window the caller narrows to cannot exclude a value
    the histogram counted.
    """
    obs_lo, obs_hi = np.inf, -np.inf
    for block in make_blocks():
        finite = _finite64(block)
        mask = (finite >= bin_lo) & ((finite <= bin_hi) if last else (finite < bin_hi))
        if mask.any():
            inside = finite[mask]
            obs_lo = min(obs_lo, float(inside.min()))
            obs_hi = max(obs_hi, float(inside.max()))
    return obs_lo, obs_hi


def _select_among_ties(make_blocks, target: int, lo: float, hi: float) -> float:
    """Select rank *target* when the window can no longer be split.

    Reached when the chosen bin *is* the whole window, which happens once
    `hi - lo` is down to about one ulp: every interior edge has rounded onto an
    endpoint, so bisecting again reproduces the same round forever. A run of
    more than `_QUANTILE_EXACT_CAP` identical values takes the search here — a
    constant background region in a full-size volume does exactly that — and
    the `lo == hi` guard does not catch it, because `hi` is the float *above*
    `lo`, not `lo`.

    An interval that narrow holds at most a couple of distinct floats (that is
    what "cannot be split" means), however many values sit on them, so counting
    per distinct value is bounded and gives the exact answer directly.

    ``-0.0`` and ``0.0`` are one dict key here, so which sign survives depends
    on which block was seen first. `stream_quantile` settles the sign of a zero
    from the data on the way out (:func:`_zero_sign`), in one place, rather
    than each selector doing it — and settling is not normalising: an
    unconditional ``+0.0`` would break numpy parity.
    """
    tally: dict[float, int] = {}
    for block in make_blocks():
        finite = _finite64(block)
        window = finite[(finite >= lo) & (finite <= hi)]
        if window.size:
            values, counts = np.unique(window, return_counts=True)
            for value, n in zip(values.tolist(), counts.tolist()):
                tally[value] = tally.get(value, 0) + n
    seen = 0
    for value in sorted(tally):
        seen += tally[value]
        if target < seen:
            return float(value)
    return float(hi)


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

    A dataset with no blocks at all — a zero-length axis, which is what a
    fully-masked or mis-selected input produces — sums to ``0.0`` rather than
    raising. The fold never runs there, so the accumulator is still the ``init``
    it started as.
    """

    def fold(acc, block):
        finite = block[np.isfinite(block)]
        return neumaier_sum(finite, state=acc)

    state = block_reduce(dset, fold, budget_bytes=budget_bytes, init=NeumaierState())
    return state.value


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
