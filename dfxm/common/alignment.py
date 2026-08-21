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
        # Generous enough that the aligned canvas — wider than the input in X
        # and denser in Z — still fits one block, so the in-core path stays one
        # pass over one array. The fast-path guard, expressed in code rather
        # than hoped for. A pathological pad that overflows it costs extra
        # passes, never a different answer.
        budget_bytes=volume.nbytes * 8 + (1 << 20),
    )
    whole = slice(0, streamed.shape[0])
    data = None
    for zsl, block in streamed.blocks():
        if data is None:
            if zsl == whole:
                # The budget above normally yields exactly one block, and that
                # block already IS the whole aligned volume — a freshly
                # allocated array that nothing else holds. Adopting it rather
                # than copying into a second one keeps this path's peak memory
                # where it was before the façade: one aligned volume, not two.
                data = block
                break
            data = np.empty(streamed.shape, dtype=streamed.dtype)
        data[zsl] = block
    if data is None:  # an empty Z axis yields no block at all
        data = np.empty(streamed.shape, dtype=streamed.dtype)
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
    """

    shape: tuple[int, int, int]
    dtype: np.dtype
    z_uniform_um: np.ndarray
    scale_z_um: float
    pad_left: int
    pad_right: int
    center_offset: float
    blocks: Callable[[], Iterator[tuple[slice, np.ndarray]]]


def _input_span(z_um: np.ndarray, z_target: np.ndarray) -> tuple[int, int]:
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

    **But the span can degenerate to the whole array**, and then
    ``dset[in_lo:in_hi]`` reads the entire input volume for that block —
    ``budget_bytes`` bounds the *output* block, so a badly interleaved samz can
    push the working set past the caller's budget. Only a non-monotonic samz
    does this (a decreasing one is fine: reversing keeps neighbours adjacent),
    and raster samz is monotone by construction, so this is a pathological-input
    caveat rather than a live risk — but it is a memory caveat, not merely a
    speed one.

    The sort is ``mergesort`` to match ``interp1d``'s own stable sort, so
    duplicated samz values keep the same layer pairing in both paths.
    """
    n = len(z_um)
    if n <= 2:
        return 0, n
    order = np.argsort(z_um, kind="mergesort")
    ordered = np.asarray(z_um)[order]
    lo_idx = (
        int(np.clip(np.searchsorted(ordered, float(np.min(z_target)), side="left"), 1, n - 1)) - 1
    )
    hi_idx = int(np.clip(np.searchsorted(ordered, float(np.max(z_target)), side="left"), 1, n - 1))
    used = order[lo_idx : hi_idx + 1]
    return int(used.min()), int(used.max()) + 1


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
    budget_bytes: int,
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
    out_step = max(1, min(nz, int(max(1, budget_bytes) // per_out_layer)))

    def _blocks(offset: float = 0.0):
        for start in range(0, nz, out_step):
            stop = min(start + out_step, nz)
            z_target = z_uniform[start:stop]
            in_lo, in_hi = _input_span(z_um, z_target)
            raw = dset[in_lo:in_hi]
            v = np.abs(raw) if take_abs else raw
            v = apply_roi_3d(v, roi_x, roi_y)
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
            yield slice(start, stop), (v - offset if offset else v)

    offset = 0.0
    if center_method:
        # The mean is one traversal, so it always streams. The median is
        # several, and re-running the alignment for each of them is the one
        # cost worth spending memory to avoid — the aligned volume goes into a
        # cache once and the quantile's passes read that instead.
        multi_pass = center_method.lower() == "median"
        fits = nz * per_out_layer <= budget_bytes
        if multi_pass and fits:
            # `fits` is exactly the condition `out_step == nz`, so the stream
            # yields ONE block and that block already IS the whole aligned
            # volume — a fresh array nothing else holds. Adopt it.
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
                offset = _center_offset(
                    lambda: volumeio.dataset_blocks(cache, budget_bytes=budget_bytes),
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
        blocks=lambda: _blocks(offset),
    )
