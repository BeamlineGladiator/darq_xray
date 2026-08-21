"""Shared volume alignment — one implementation, voxel-identical everywhere.

Ported from the alignment used by the PVTI exporter and the aligned-volume
viewer so that visualization, ParaView export, oblique slices and the rocking
volumes all co-register. The fixed pipeline (order matters) is::

    abs (FWHM only) -> ROI -> samy sub-pixel X-shift (canvas auto-expand)
        -> Z interpolation onto a uniform grid -> centre (CoM / strain only)

All geometry constants (pixel scale, samy direction, ROI, darfix origin) are
explicit arguments here rather than module globals, so the same code serves any
experiment.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import shift as ndi_shift

from dfxm.common import volumeio


def apply_roi_3d(data: np.ndarray, roi_x: tuple | None, roi_y: tuple | None) -> np.ndarray:
    """Crop a (Z, Y, X) volume in pixel coordinates; ``None`` keeps the full axis."""
    _z, y, x = data.shape
    xs, xe = (roi_x[0], roi_x[1]) if roi_x else (0, x)
    ys, ye = (roi_y[0], roi_y[1]) if roi_y else (0, y)
    return data[:, ys:ye, xs:xe]


def _samy_ref(samy: np.ndarray, ref_samy: float | None) -> float:
    """Resolve the samy reference: explicit value, else the first layer."""
    return float(samy[0]) if ref_samy is None else float(ref_samy)


def compute_pad_left(
    samy: np.ndarray, scale_x: float, samy_direction: int = 1, ref_samy: float | None = None
) -> int:
    """Left X-padding (px) that :func:`apply_samy_shifts_to_volume` will add.

    Same formula as the shift, so a world origin can be anchored to the
    reference (un-shifted) column. ``ref_samy`` overrides the default
    first-layer reference (the rocking stage anchors to the mosa frame).
    """
    if samy is None or len(samy) == 0:
        return 0
    offsets_px = samy_direction * (np.asarray(samy) - _samy_ref(samy, ref_samy)) * 1000.0 / scale_x
    return max(0, int(np.ceil(-np.min(offsets_px))))


def compute_pad_right(
    samy: np.ndarray, scale_x: float, samy_direction: int = 1, ref_samy: float | None = None
) -> int:
    """Right X-padding (px) that :func:`apply_samy_shifts_to_volume` will add."""
    if samy is None or len(samy) == 0:
        return 0
    offsets_px = samy_direction * (np.asarray(samy) - _samy_ref(samy, ref_samy)) * 1000.0 / scale_x
    return max(0, int(np.ceil(np.max(offsets_px))))


def apply_samy_shifts_to_volume(
    volume: np.ndarray,
    samy: np.ndarray,
    scale_x: float,
    samy_direction: int = 1,
    ref_samy: float | None = None,
    *,
    pad: tuple[int, int] | None = None,
) -> np.ndarray:
    """Shift each Z-layer along image-X by its samy offset, expanding the canvas.

    samy is in mm; offsets are relative to ``ref_samy`` (default: the first
    layer). The canvas grows so nothing is clipped, and exposed regions are
    NaN-padded. Pass ``ref_samy`` to anchor to an external frame (e.g. the
    rocking volume to the mosa reference column).

    ``pad`` overrides the ``(left, right)`` canvas growth. It exists for
    block-wise callers: the pad is a property of the WHOLE volume's samy
    range, so deriving it from a block's slice would give each block a
    different width. ``None`` computes it from *samy*, as every in-core
    caller wants.
    """
    n_layers = volume.shape[0]
    n_use = n_layers if len(samy) == n_layers else min(n_layers, len(samy))

    ref = _samy_ref(samy, ref_samy)
    samy_offsets_px = samy_direction * (np.asarray(samy[:n_use]) - ref) * 1000.0 / scale_x
    if pad is None:
        pad_left = max(0, int(np.ceil(-np.min(samy_offsets_px))))
        pad_right = max(0, int(np.ceil(np.max(samy_offsets_px))))
    else:
        pad_left, pad_right = int(pad[0]), int(pad[1])

    ny, nx_orig = volume.shape[1], volume.shape[2]
    nx_new = nx_orig + pad_left + pad_right
    shifted = np.full((n_layers, ny, nx_new), np.nan, dtype=volume.dtype)

    for i in range(n_layers):
        padded = np.full((ny, nx_new), np.nan, dtype=volume.dtype)
        padded[:, pad_left : pad_left + nx_orig] = volume[i]
        if i < n_use and abs(samy_offsets_px[i]) > 0.01:
            padded = ndi_shift(
                padded, shift=(0.0, samy_offsets_px[i]), order=1, mode="constant", cval=np.nan
            )
        shifted[i] = padded
    return shifted


def _z_positions_um(samz: np.ndarray, n_layers: int, ref_samz: float | None) -> np.ndarray:
    """Layer Z positions in µm relative to ``ref_samz`` (default: the first)."""
    n_use = min(n_layers, len(samz))
    ref = float(samz[0]) if ref_samz is None else float(ref_samz)
    return (np.asarray(samz[:n_use]) - ref) * 1000.0


def _median_z_step(z_um: np.ndarray) -> float:
    """Median |ΔZ| in µm, floored away from zero so a scale is never 0."""
    step = float(np.median(np.abs(np.diff(z_um)))) if len(z_um) > 1 else 1.0
    return 1.0 if step < 1e-6 else step


def _z_grid(
    samz: np.ndarray, n_layers: int, ref_samz: float | None
) -> tuple[np.ndarray, np.ndarray, float]:
    """The uniform Z grid and its step, from motor positions alone.

    Returns ``(z_um, z_uniform_um, scale_z_um)``. Needing no voxel is the point:
    a streaming caller sizes its output from this before reading anything, and
    :func:`interpolate_to_uniform_z` derives its own grid here too, so the two
    cannot drift apart.
    """
    z_um = _z_positions_um(samz, n_layers, ref_samz)
    median_step = _median_z_step(z_um)
    if len(z_um) <= 1:
        return z_um, z_um, median_step
    z_min, z_max = float(z_um.min()), float(z_um.max())
    n_uniform = max(2, int(np.round((z_max - z_min) / median_step)) + 1)
    z_uniform = np.linspace(z_min, z_max, n_uniform)
    scale_z = float(z_uniform[1] - z_uniform[0]) if n_uniform > 1 else median_step
    return z_um, z_uniform, scale_z


def interpolate_to_uniform_z(
    volume: np.ndarray,
    samz: np.ndarray,
    ref_samz: float | None = None,
    *,
    z_uniform: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample irregular samz (mm) layers onto a uniform Z grid (µm).

    Z origin is ``ref_samz`` (default: the first layer); pass it to anchor the
    rocking volume to the mosa Z reference. ``z_uniform`` supplies the target
    grid instead of deriving it from *samz* — block-wise callers pass the
    global grid's own sub-range, since a block's samz would imply a different
    grid. Returns ``(interp_volume, z_uniform_um, scale_z_um)``.
    """
    n_use = min(volume.shape[0], len(samz))
    if z_uniform is None:
        z_um, z_uniform, scale_z = _z_grid(samz, volume.shape[0], ref_samz)
    else:
        z_um = _z_positions_um(samz, volume.shape[0], ref_samz)
        z_uniform = np.asarray(z_uniform, dtype=float)
        scale_z = float(z_uniform[1] - z_uniform[0]) if len(z_uniform) > 1 else _median_z_step(z_um)
    n_uniform = len(z_uniform)

    if n_use <= 1 and n_uniform <= 1:
        # A single layer has no Z extent: return it unchanged on a length-1 grid
        # with a nonzero scale (avoids scale_z=0 and the all-NaN interp1d
        # result). A block-wise caller reaches this with an explicit length-1
        # grid, for the same reason and with the same answer.
        return volume[:1].astype(volume.dtype), z_uniform, scale_z

    vol_use = volume[:n_use]
    ny, nx = vol_use.shape[1], vol_use.shape[2]
    flat = vol_use.reshape(n_use, -1)
    interp = interp1d(z_um, flat, axis=0, kind="linear", bounds_error=False, fill_value=np.nan)
    vol_interp = interp(z_uniform).reshape(n_uniform, ny, nx)
    return vol_interp, z_uniform, scale_z


def _center_offset(blocks_factory, method: str) -> float:
    """The centring statistic over finite voxels, as one shared definition.

    In-core and streamed centring must agree bit-for-bit, so both come through
    here: :func:`volumeio.stream_mean` and :func:`volumeio.stream_quantile`
    rather than ``np.nanmean``/``np.nanmedian``. The compensated mean differs
    from ``np.nanmean`` by a few ulps — that is the drift the design accepted,
    and taking it in both paths is what makes "identical for any budget"
    exactly true instead of nearly true. The drift looks larger in ulps the
    closer the mean sits to zero, which for a volume about to be centred is
    very close indeed: 41 ulps on a 13.5 M-voxel volume, where the compensated
    value is the exactly-rounded mean and numpy's pairwise order is the one
    that is off. The median is bit-equal to ``np.nanmedian``; only the mean
    moves.

    Cost, measured: :func:`volumeio.neumaier_sum` is vectorised over lanes, so
    the mean of 13.5 M finite voxels costs ~0.09 s — faster than ``np.nansum``,
    and 23× the ~2.2 s the per-element loop it replaced cost. Centring is
    therefore dominated by the second pass over the aligned volume that
    subtracting an offset forces, not by the statistic:
    ``align_volume(center_method="mean")`` on a 60×300×400 volume measures
    0.84 s against 0.38 s uncentred (it was 1.28 s before the vectorisation).

    *blocks_factory* is a zero-argument callable returning a fresh iterable of
    arrays, because the quantile traverses more than once.
    """
    m = method.lower()
    if m == "mean":
        return float(volumeio.stream_mean(blocks_factory()))
    if m == "median":
        return float(volumeio.stream_quantile(blocks_factory, 50.0))
    raise ValueError(f"center_around_zero: unknown method {method!r} (expected mean/median)")


def center_around_zero(data: np.ndarray, method: str = "mean") -> tuple[np.ndarray, float]:
    """Subtract a global statistic over finite voxels. Returns ``(data, offset)``."""
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        # Validate the method even when there is nothing to centre, so a typo
        # is still an error on an all-NaN volume.
        _center_offset(lambda: [np.empty(0)], method)
        return data, 0.0
    offset = _center_offset(lambda: [valid], method)
    return data - offset, offset


def raw_detector_origin(
    samy: np.ndarray,
    z_positions: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    roi_x: tuple | None = None,
    roi_y: tuple | None = None,
    darfix_origin_xy: tuple = (0, 0),
    samy_direction: int = 1,
) -> tuple[float, float, float]:
    """World origin (µm) placing this volume in the raw-detector-absolute frame.

    A padded voxel at column p, row j maps back to raw pixel
    ``(p - pad_left + roi_x0 + darfix_x0, j + roi_y0 + darfix_y0)``; the origin
    is that mapping times the pixel size, with Z taken from ``z_positions[0]``.
    """
    pad_left = compute_pad_left(samy, scale_x, samy_direction)
    dx0, dy0 = darfix_origin_xy
    roi_x0 = roi_x[0] if roi_x else 0
    roi_y0 = roi_y[0] if roi_y else 0
    ox = (roi_x0 + dx0 - pad_left) * scale_x
    oy = (roi_y0 + dy0) * scale_y
    oz = float(z_positions[0]) if z_positions is not None and len(z_positions) else 0.0
    return (ox, oy, oz)


@dataclass
class AlignedVolume:
    """Output of :func:`align_volume`."""

    data: np.ndarray
    z_uniform_um: np.ndarray
    scale_z_um: float
    pad_left: int
    center_offset: float = 0.0


def materialise_blocks(blocks: Callable[[], Iterator[tuple[slice, np.ndarray]]], shape, dtype):
    """Drain a block factory into one ``shape``-sized array.

    **Adopts a single covering block rather than copying it.** A budget generous
    enough to leave the stream one block hands over an array that already *is*
    the whole volume and that nothing else holds; copying it into a fresh
    ``np.empty`` would hold two whole volumes at once, which is the peak this
    would be trying to avoid. An empty Z axis yields no block at all and gets the
    empty array its shape describes.

    One implementation, because four callers need this seam and each of them
    would get the adoption wrong in the same expensive way:
    :func:`align_volume` (the in-core façade), ``slices._materialise``,
    ``paraview._drained`` (each stage's in-core rung) and
    ``visualize._LayerSource.whole`` (the 3-D consumer that cannot stream).
    """
    covering = slice(0, int(shape[0]))
    data = None
    for zsl, block in blocks():
        if data is None:
            if zsl == covering:
                return block
            data = np.empty(tuple(shape), dtype=dtype)
        data[zsl] = block
    return np.empty(tuple(shape), dtype=dtype) if data is None else data


def align_volume(
    volume: np.ndarray,
    samy: np.ndarray,
    samz: np.ndarray,
    *,
    scale_x: float,
    samy_direction: int = 1,
    roi_x: tuple | None = None,
    roi_y: tuple | None = None,
    take_abs: bool = False,
    center_method: str | None = None,
) -> AlignedVolume:
    """Run the full fixed alignment pipeline on a (Z, Y, X) volume.

    ``take_abs`` is for FWHM maps (drop unphysical negative fits, NaN-safe);
    ``center_method`` ("mean"/"median") centres CoM/strain volumes around zero.

    This is the in-core façade over :func:`align_volume_streamed`: it drains
    the stream into one array. Both paths therefore run identical arithmetic —
    there is one implementation, not two that must be kept in step.
    """
    streamed = align_volume_streamed(
        volume,
        samy,
        samz,
        scale_x=scale_x,
        samy_direction=samy_direction,
        roi_x=roi_x,
        roi_y=roi_y,
        take_abs=take_abs,
        center_method=center_method,
        # `None` is "one block, whatever it costs" — the honest spelling of what
        # this path needs. It used to ask for `volume.nbytes * 8`, a guess at a
        # budget big enough to leave one block; the moment `budget_bytes` came
        # to mean the whole working set rather than the block, that guess was no
        # longer big enough, and a caller that materialises the entire volume
        # anyway has no business pretending to a budget.
        budget_bytes=None,
    )
    # The budget above normally yields exactly one block, and that block already
    # IS the whole aligned volume — a freshly allocated array nothing else holds.
    # `materialise_blocks` adopts it rather than copying into a second one, which
    # keeps this path's peak memory where it was before the façade: one aligned
    # volume, not two.
    data = materialise_blocks(streamed.blocks, streamed.shape, streamed.dtype)
    return AlignedVolume(
        data, streamed.z_uniform_um, streamed.scale_z_um, streamed.pad_left, streamed.center_offset
    )


@dataclass
class StreamedAlignment:
    """The aligned volume as a re-traversable stream of Z-blocks.

    Every field but ``blocks`` is known before a voxel is read — they come from
    the small 1-D motor arrays and the dataset's shape — so a writer can size
    its output up front. ``blocks`` is a factory: call it to get a fresh
    iterator. Traversing twice re-runs the alignment chain, which is why
    consumers that need several passes should say so.

    ``block_layers`` and ``working_set_bytes`` report the blocking that
    :func:`align_volume_streamed` solved for: how many output layers a block
    carries, and the peak the working-set model says producing one costs. The
    second is the number ``budget_bytes`` was checked against, so a caller can
    see what it actually bought — and see it *before* reading a voxel.
    """

    shape: tuple[int, int, int]
    dtype: np.dtype
    z_uniform_um: np.ndarray
    scale_z_um: float
    pad_left: int
    pad_right: int
    center_offset: float
    block_layers: int
    working_set_bytes: int
    blocks: Callable[[], Iterator[tuple[slice, np.ndarray]]]


@dataclass(frozen=True)
class _WorkingSet:
    """Bytes the streamed chain holds **at once** while producing one block.

    Counted from the body of ``_blocks`` and the functions it calls, not fitted
    to a measurement. Every coefficient is per *layer*, so the model is linear
    in the blocking and can be solved for it:

    ``read_layer`` (per INPUT layer)
        ``raw = dset[in_lo:in_hi]``. An h5py dataset materialises the span; a
        plain ``ndarray`` returns a view, which costs nothing. ``take_abs``
        adds ``np.abs(raw)`` on top, and ``raw`` stays bound in the generator
        frame for the whole iteration, so with both the span is held twice.

    ``canvas_layer`` (per INPUT layer) x **3**
        The X-padded samy canvas (``shifted`` in
        :func:`apply_samy_shifts_to_volume`), then ``interp1d(copy=True)``'s own
        copy of it, then the permuted copy ``np.take(y, ind, axis=axis)`` builds
        while that copy is still alive. Three, **always**: scipy 1.11.4's
        ``interp1d.__init__`` runs ``ind = argsort(x); y = np.take(y, ind, axis)``
        unconditionally — :func:`interpolate_to_uniform_z` never passes
        ``assume_sorted``, so an ascending samz takes the sort branch exactly
        like a shuffled one. This was briefly conditioned on the samz being
        unsorted, which contradicted the code it cited and survived only because
        the per-output-layer terms usually dominate the per-input-layer ones; on
        an ascending samz with bimodal Z spacing they do not, and the model then
        under-priced the peak.

    ``interp_layer`` (per OUTPUT layer)
        ``scipy``'s ``_call_linear`` holds five block-sized arrays at its peak:
        ``y_lo`` and ``y_hi`` (fancy-indexed out of the input dtype) plus
        ``slope``, the ``slope * dx`` product and ``y_new`` (float64, because
        the Z coordinates are float64). Hence
        ``ny * nx * (2 * in_itemsize + 3 * out_itemsize)``.

    ``carry_layer`` (per OUTPUT layer)
        The block the consumer is still holding. ``for _sl, block in
        streamed.blocks()`` keeps the previous block bound across the ``next()``
        that computes the following one, so two blocks coexist — every consumer
        in this repo iterates that way, so it is a real term, not a caller's
        mistake.

    ``stat_layer`` (per OUTPUT layer)
        What a centring pass leaves bound across that same ``next()``.
        ``stream_mean`` keeps ``finite``; ``volumeio._select_rank`` keeps
        ``finite``, ``window`` and the ``searchsorted`` indices — three
        float64/int64 arrays of the block's element count, plus the
        ``isfinite`` mask. So the interpolator's temporaries for block *i+1*
        and the statistic's leftovers from block *i* are live together, which
        is precisely where the median path was measured to peak. Zero when
        there is no centring; this term also covers the ``v - offset``
        temporary the centred stream yields through.

    ``fixed``
        Blocking-independent, and paid on **every** candidate blocking: the
        per-layer ``padded`` array and :func:`scipy.ndimage.shift`'s output
        inside the samy loop, plus — when centring —
        :func:`volumeio.centring_scaffold_bytes`'s "during pass" figure (the
        Neumaier lanes for the mean; the histogram edges, counts and the
        accumulating survivors for the median).

    ``floor``
        A peak **no blocking can reduce**, taken as a maximum against everything
        above rather than added to it: the quantile's survivors held three times
        over while they are concatenated and sorted, which happens after the
        traversal, when the chain's own temporaries are gone. A budget below
        this cannot be met at any block size, which is what makes it a floor and
        not a term.

    These two were once dismissed in this docstring as "about a megabyte… a
    budget in the megabytes swallows them". That was wrong, and measured wrong:
    a 40x64x80 median at a 1.66 MB budget peaked at **1.90x** it, and at 0.61 MB
    at **3.63x**, because the histogram scaffolding alone is 1.6 MB. Constants
    that no blocking can pay off still have to be *in* the budget or the budget
    is not one.

    Still outside the model: whatever the *consumer* accumulates — a caller that
    writes every block into one in-core array (:func:`align_volume`) is
    allocating the whole output by choice, and that is its own budget to keep.
    Also outside: memory that never passes through the Python allocator, since
    every figure here is a ``tracemalloc`` one — h5py's chunk cache, a memmap's
    resident pages and allocator fragmentation are RSS this does not see.
    """

    read_layer: int
    canvas_layer: int
    canvas_copies: int
    interp_layer: int
    carry_layer: int
    stat_layer: int
    fixed: int
    floor: int

    def bytes_for(self, n_in: int, n_out: int) -> int:
        """Peak bytes for a block of *n_out* output layers off *n_in* input layers."""
        return max(
            n_in * (self.read_layer + self.canvas_copies * self.canvas_layer)
            + n_out * (self.interp_layer + self.carry_layer + self.stat_layer)
            + self.fixed,
            self.floor,
        )


def _sorted_z(z_um: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The mergesort order of *z_um* and the sorted values, or ``None`` if unused.

    Hoisted out of :func:`_input_span` so that solving the blocking — which
    asks for the span of every candidate block — sorts once instead of once
    per question.
    """
    if len(z_um) <= 2:
        return None
    order = np.argsort(z_um, kind="mergesort")
    return order, np.asarray(z_um)[order]


def _input_span(
    z_um: np.ndarray,
    z_target: np.ndarray,
    *,
    sorted_z: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[int, int]:
    """Input layer indices whose values linear interpolation of *z_target* reads.

    Returns a half-open ``(start, stop)`` in the ORIGINAL layer order, because
    that is how the block is read off disk.

    ``interp1d`` brackets each target with the pair ``(idx - 1, idx)``, where
    ``idx = clip(searchsorted(sorted_x, target, side="left"), 1, n - 1)`` — the
    same expression scipy's ``_call_linear`` uses. ``searchsorted`` is monotone
    in the target, so the whole block's needs are bounded by its smallest and
    largest target. Note the ``- 1``: a target sitting exactly ON a sample node
    is still interpolated from the pair *below* it, so the node alone is not
    enough and a naive span would hand ``interp1d`` a single point.

    A decreasing or unsorted samz is handled by working in the sorted order and
    then spanning ``[min, max]`` of the original indices that order picked out.
    The answer stays correct however scrambled samz is, because adding input
    layers cannot loosen a bracket: the sub-range always contains the pair the
    full array would have used, and no layer lies between that pair.

    **The span can degenerate to the whole array**: a badly interleaved samz
    puts a block's Z neighbours far apart in file order, and then
    ``dset[in_lo:in_hi]`` reads the entire input volume for one output block.
    That is not a budget hole — :func:`align_volume_streamed` asks this function
    for the span of every candidate blocking *before* it reads anything, so the
    degenerate span is priced into the block size and the budget still holds; it
    simply buys fewer output layers per block. Only a non-monotonic samz does it
    (a decreasing one is fine: reversing keeps neighbours adjacent), and raster
    samz is monotone by construction.

    The sort is ``mergesort`` to match ``interp1d``'s own stable sort, so
    duplicated samz values keep the same layer pairing in both paths.
    *sorted_z* passes in :func:`_sorted_z`'s result, so a caller asking about
    many candidate blocks does not re-sort per question.
    """
    n = len(z_um)
    if n <= 2:
        return 0, n
    order, ordered = _sorted_z(z_um) if sorted_z is None else sorted_z
    lo_idx = (
        int(np.clip(np.searchsorted(ordered, float(np.min(z_target)), side="left"), 1, n - 1)) - 1
    )
    hi_idx = int(np.clip(np.searchsorted(ordered, float(np.max(z_target)), side="left"), 1, n - 1))
    used = order[lo_idx : hi_idx + 1]
    return int(used.min()), int(used.max()) + 1


def _max_input_span(
    z_um: np.ndarray,
    z_uniform: np.ndarray,
    step: int,
    sorted_z: tuple[np.ndarray, np.ndarray] | None,
) -> int:
    """The widest input span any block takes when the stream steps by *step*.

    The blocking is fixed (``range(0, nz, step)``), so this is the exact number
    of input layers the worst block of that blocking will read — not a bound on
    it. That exactness is what keeps the budget honest on a non-monotonic samz,
    where the span is nothing like the block.
    """
    nz = len(z_uniform)
    widest = 0
    for start in range(0, nz, step):
        lo, hi = _input_span(z_um, z_uniform[start : start + step], sorted_z=sorted_z)
        widest = max(widest, hi - lo)
    return widest


def _solve_out_step(
    nz: int,
    z_um: np.ndarray,
    z_uniform: np.ndarray,
    cost: _WorkingSet,
    budget_bytes: int | None,
    sorted_z: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[int, int]:
    """Largest block (in output layers) whose modelled working set fits the budget.

    Returns ``(out_step, working_set_bytes)``. ``budget_bytes=None`` means "one
    block, whatever it costs" — the in-core façade's request.

    Bisection, not division: the working set is linear in the block size *and*
    in the input span the block reads, and the second is a step function of the
    first that only :func:`_max_input_span` can answer. Every step the search
    accepts has been priced, so a non-monotonic span can only make the answer
    conservative, never wrong — a bisection step is adopted solely on a
    ``fits`` it measured.

    The floor is one output layer. Progress beats precision: a block that
    overruns the budget still runs, and the overrun is reported (a warning, and
    ``StreamedAlignment.working_set_bytes``) rather than raised, because the
    alternative is a stage that cannot run at all on a machine that is merely
    tight.
    """

    def ws(step: int) -> int:
        return cost.bytes_for(_max_input_span(z_um, z_uniform, step, sorted_z), step)

    if nz <= 0:
        return 1, cost.bytes_for(0, 0)
    if budget_bytes is None:
        return nz, ws(nz)
    budget = max(1, int(budget_bytes))
    whole = ws(nz)
    if whole <= budget:
        # The generous case, and the common one — answered with a single span
        # query instead of the O(nz) one that pricing a one-layer block costs.
        return nz, whole
    best, best_bytes = 1, ws(1)
    lo, hi = 2, nz - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        mid_bytes = ws(mid)
        if mid_bytes <= budget:
            best, best_bytes = mid, mid_bytes
            lo = mid + 1
        else:
            hi = mid - 1
    return best, best_bytes


def align_volume_streamed(
    dset,
    samy: np.ndarray,
    samz: np.ndarray,
    *,
    scale_x: float,
    samy_direction: int = 1,
    roi_x: tuple | None = None,
    roi_y: tuple | None = None,
    take_abs: bool = False,
    center_method: str | None = None,
    budget_bytes: int | None,
    scratch_dir: str | None = None,
) -> StreamedAlignment:
    """The fixed alignment pipeline, streamed in Z-blocks.

    Runs exactly the same steps in exactly the same order as
    :func:`align_volume` — ``abs`` (FWHM only), ROI, samy X-shift, uniform-Z
    interpolation, centring — by calling the same functions on blocks. The
    global quantities those steps would otherwise derive per block (the samy
    canvas pad, the Z grid) are computed once from the motor arrays and passed
    in explicitly. That is the whole trick: a block's own samy implies a
    narrower canvas and a block's own samz a different grid, so each block
    would land on a frame of its own.

    *dset* is anything that slices like ``dset[a:b]`` and has ``shape`` and
    ``dtype`` — an h5py dataset, a memmap or an in-memory array.

    ``budget_bytes`` is a bound on the **working set**: everything the chain
    holds at once to produce a block, priced by :class:`_WorkingSet` — the input
    span the block reads, the padded samy canvas and ``interp1d``'s copy of it,
    the interpolator's five block-sized temporaries, and the block the consumer
    is still carrying. It is *not* a bound on the block alone; sizing blocks by
    the block's own bytes (what this function did until the working-set model
    replaced it) overshot the real peak by 3.6x to 7.7x, worse the tighter the
    budget, because the span and the fixed temporaries do not shrink with the
    block. A caller handing over the machine's advised headroom gets a run that
    fits in it. ``None`` means "one block, whatever it costs" — what the in-core
    façade wants, since it is materialising the whole volume anyway. See
    :class:`_WorkingSet` for the constant ~1 MB the model deliberately excludes.

    The floor is one output layer. If even that overruns *budget_bytes* the run
    proceeds and warns: on this pipeline a slow stage beats a stage that refuses
    to start.

    Centring costs an extra traversal, because the statistic is over the
    *aligned* volume — NaN-padded canvas, interpolated Z grid — and cannot be
    precomputed from the source. ``mean`` is exactly one extra pass. ``median``
    traverses four or more, so the aligned volume is cached once and those
    passes read the cache: in RAM when it fits ``budget_bytes`` (which is the
    in-core façade's path), else in ``scratch_dir``. With neither, it falls back
    to re-running the alignment per pass — slower, never failed.
    """
    n_layers = int(dset.shape[0])
    z_um, z_uniform, scale_z = _z_grid(samz, n_layers, None)
    pad_left = compute_pad_left(samy, scale_x, samy_direction)
    pad_right = compute_pad_right(samy, scale_x, samy_direction)

    _z, y, x = dset.shape
    xs, xe = (roi_x[0], roi_x[1]) if roi_x else (0, x)
    ys, ye = (roi_y[0], roi_y[1]) if roi_y else (0, y)
    ny = ye - ys
    nx = (xe - xs) + pad_left + pad_right
    nz = len(z_uniform)
    shape = (nz, ny, nx)
    # The chain upcasts to float64 in `interpolate_to_uniform_z` (interp1d's
    # float64 arithmetic), except on the single-layer early return, which hands
    # the layer back in its own dtype. Declaring what the blocks will actually
    # be keeps the in-core facade's output dtype exactly what it was.
    dtype = np.dtype(np.float64) if len(z_um) > 1 else np.dtype(dset.dtype)

    per_out_layer = max(1, ny * nx * dtype.itemsize)
    in_item = np.dtype(dset.dtype).itemsize
    # Basic slicing of a real ndarray is a view, so `dset[in_lo:in_hi]` costs
    # nothing there; an h5py dataset (and anything else duck-typed) materialises
    # the span. A memmap is an ndarray subclass but its slice is backed by pages
    # that do become resident, so it is charged like a read.
    reads_into_ram = not isinstance(dset, np.ndarray) or isinstance(dset, np.memmap)
    per_in_layer = y * x * in_item
    # How many float64/int64 arrays of the block's element count the centring
    # pass is still holding when the generator computes the next block: none
    # without centring, `finite` for the mean, `finite` + `window` + the
    # searchsorted indices for the median's rank search. The +1 is the
    # `isfinite` boolean mask.
    # An unknown method is priced at the worst case rather than rejected here:
    # `_center_offset` is the single place that validates it, and it raises.
    retained = {None: 0, "mean": 1}.get(center_method.lower() if center_method else None, 3)
    # The reductions' own blocking-independent cost, priced by the module that
    # allocates it. `median` is the fallback for an unvalidated method, matching
    # `retained` above — it is the more expensive of the two, and
    # `_center_offset` still raises on anything that is neither.
    scaffold, floor = (
        volumeio.centring_scaffold_bytes("mean" if retained == 1 else "median", nz * ny * nx)
        if center_method
        else (0, 0)
    )
    cost = _WorkingSet(
        read_layer=per_in_layer * (int(reads_into_ram) + int(bool(take_abs))),
        canvas_layer=ny * nx * in_item,
        canvas_copies=3,
        interp_layer=ny * nx * (2 * in_item + 3 * dtype.itemsize),
        carry_layer=per_out_layer,
        stat_layer=ny * nx * (8 * retained + 1) if retained else 0,
        fixed=2 * ny * nx * in_item + scaffold,
        floor=floor,
    )
    # One mergesort for the whole call: the solver asks for the span of every
    # candidate blocking, and `_blocks` asks again for every block it yields.
    sorted_z = _sorted_z(z_um)
    out_step, working_set = _solve_out_step(nz, z_um, z_uniform, cost, budget_bytes, sorted_z)
    if budget_bytes is not None and working_set > budget_bytes:
        warnings.warn(
            f"align_volume_streamed: the smallest working set this run can have "
            f"is about {working_set} B, over the {int(budget_bytes)} B budget "
            "(one output layer plus the blocking-independent scaffolding); "
            "running anyway at one layer per block",
            stacklevel=2,
        )

    def _blocks(offset: float = 0.0):
        for start in range(0, nz, out_step):
            stop = min(start + out_step, nz)
            z_target = z_uniform[start:stop]
            in_lo, in_hi = _input_span(z_um, z_target, sorted_z=sorted_z)
            raw = dset[in_lo:in_hi]
            v = np.abs(raw) if take_abs else raw
            v = apply_roi_3d(v, roi_x, roi_y)
            # A generator frame stays alive between `yield`s, so anything still
            # bound here is retained for as long as the consumer holds the
            # block. That is invisible with one stream and expensive with
            # several: paraview keeps one open per field, and four suspended
            # frames each pinning their own input span cost four whole raw
            # volumes on top of the four blocks (measured: 144 MB of a 476 MB
            # peak on a four-field 128x192x192 export). `apply_roi_3d` returns
            # a VIEW, so this frees nothing until the samy shift or the
            # interpolation allocates and `v` stops referring to the read —
            # which is exactly when it should.
            del raw
            if samy is not None and len(samy) > 0:
                v = apply_samy_shifts_to_volume(
                    v,
                    samy[in_lo:in_hi],
                    scale_x,
                    samy_direction,
                    ref_samy=_samy_ref(samy, None),
                    pad=(pad_left, pad_right),
                )
            v, _zu, _sz = interpolate_to_uniform_z(
                v,
                samz[in_lo:in_hi],
                ref_samz=float(samz[0]),
                z_uniform=z_target,
            )
            block = v - offset if offset else v
            del v  # same reason: `v` is a second reference only when centring
            yield slice(start, stop), block
            del block  # and the block itself, once the consumer has taken it

    offset = 0.0
    if center_method:
        # The mean is one traversal, so it always streams. The median is
        # several, and re-running the alignment for each of them is the one
        # cost worth spending memory to avoid — the aligned volume goes into a
        # cache once and the quantile's passes read that instead.
        multi_pass = center_method.lower() == "median"
        fits = out_step >= nz
        if multi_pass and fits:
            # `fits` IS the condition `out_step == nz` — asked of the solved
            # blocking rather than recomputed from the output bytes, which is
            # what it used to be and which no longer means the same thing now
            # that the block is sized against the whole working set. The stream
            # therefore yields ONE block and that block already IS the whole
            # aligned volume — a fresh array nothing else holds. Adopt it.
            #
            # Copying it into a `np.empty(shape)` cache instead (the obvious
            # spelling) would hold two full-size volumes at once: measured
            # +43% peak on a 40x200x260 float32 volume, and — worse — it
            # doubles the very budget that made `fits` true, so a Wave 3 stage
            # handed the machine's advised RAM would allocate twice it and die
            # on precisely the volume this branch was meant to make cheap.
            cache = [block for _zsl, block in _blocks()]
            offset = _center_offset(lambda: cache, center_method)
            del cache
        elif multi_pass and scratch_dir is not None:
            with volumeio.scratch_array(shape, dtype, dirpath=scratch_dir) as cache:
                for zsl, block in _blocks():
                    cache[zsl] = block
                # Reading the cache back is a plain slice of a memmap, so none
                # of the ALIGNMENT chain's multipliers apply — but the
                # CONSUMER's do, and `dataset_blocks` sizes a block by the
                # block's own bytes. The quantile pass reading these blocks
                # holds exactly what the model charges as `stat_layer`
                # (`finite`, `window` and the searchsorted/clip indices —
                # float64/int64 arrays of the block's element count — plus the
                # `isfinite` mask) alongside the block itself, and the
                # blocking-independent `scaffold` on top of that. Handing
                # `budget_bytes` over raw priced none of it: measured 1.68x the
                # budget at budget/4, 3.36x at /8 and 5.20x at /128 — rising as
                # the budget shrank, the same signature the working-set model
                # was introduced to remove, and on the branch a 17 GB volume
                # actually takes. So convert the budget into the block bytes
                # that leave room for the rest, taking the scaffolding off the
                # top the way `_WorkingSet.fixed` does. Flooring at one layer
                # (via `dataset_blocks`) keeps the project's rule: a budget too
                # small to meet still runs. `budget_bytes` is never None on this
                # branch (`None` makes `fits` true, taking the one above).
                #
                # Per element of a cache block, at the quantile pass's peak:
                # `finite` and `window` (float64), the `searchsorted(...) - 1`
                # indices and `np.clip`'s separate output (int64), plus the
                # `isfinite` mask. One lane more than `stat_layer`'s `retained`,
                # because `stat_layer` prices what survives ACROSS the
                # generator's `next()` and this pass has no generator to hand
                # back to — `clip`'s input and output are live together here.
                per_element = dtype.itemsize + 8 * (retained + 1) + 1
                cache_budget = max(
                    1, (max(1, int(budget_bytes) - scaffold) * dtype.itemsize) // per_element
                )
                offset = _center_offset(
                    lambda: volumeio.dataset_blocks(cache, budget_bytes=cache_budget),
                    center_method,
                )
        else:
            # Either a single-pass statistic, or a median with nowhere to cache:
            # re-run the chain rather than fail, per the project's governing rule.
            offset = _center_offset(lambda: (b for _sl, b in _blocks()), center_method)
        if np.isnan(offset):
            # No finite voxel anywhere, which is what makes the statistic NaN.
            # `center_around_zero` leaves such a volume alone and reports 0.0;
            # match it, rather than turning every voxel into NaN. An infinite
            # offset is NOT caught here: that one is a real statistic, and the
            # in-core path subtracts it too.
            offset = 0.0

    return StreamedAlignment(
        shape=shape,
        dtype=dtype,
        z_uniform_um=z_uniform,
        scale_z_um=scale_z,
        pad_left=pad_left,
        pad_right=pad_right,
        center_offset=float(offset),
        block_layers=out_step,
        working_set_bytes=working_set,
        blocks=lambda: _blocks(offset),
    )
