"""Tests for dfxm.stages.matched — (samy,samz) matching, background-subtracted
frame loading, and pixel-aligned grayscale layer output.
"""

from __future__ import annotations

import os
from dataclasses import asdict

import h5py
import numpy as np
import pytest

from dfxm.common.plotting import PlotStyle
from dfxm.stages import matched as M

NF, H, W = 4, 6, 8


def _write_strain(root, name, samy, samz):
    folder = os.path.join(root, name)
    os.makedirs(folder, exist_ok=True)
    with h5py.File(os.path.join(folder, name + ".h5"), "w") as f:
        f.create_dataset("1.1/instrument/positioners/samy", data=samy)
        f.create_dataset("1.1/instrument/positioners/samz", data=samz)


def _write_rocking(root, name, samy, samz, frames):
    folder = os.path.join(root, name)
    os.makedirs(folder, exist_ok=True)
    with h5py.File(os.path.join(folder, name + ".h5"), "w") as f:
        f.create_dataset("1.1/instrument/positioners/samy", data=samy)
        f.create_dataset("1.1/instrument/positioners/samz", data=samz)
        f.create_dataset("1.1/measurement/pco_ff", data=frames.astype(np.float32))


# -- units --------------------------------------------------------------------
def test_match_nearest_threshold():
    sy = np.array([0.0, 0.002, 0.004])
    sz = np.array([0.0, 0.001, 0.003])
    ry = np.array([0.0039, 0.5])
    rz = np.array([0.0029, 0.5])
    matches, max_dist = M.match_nearest(sy, sz, ry, rz, threshold_mm=0.001)
    assert matches[2] == 0  # strain layer 2 matches rocking 0 (within threshold)
    assert matches[0] is None  # too far -> no match
    assert max_dist <= 0.001


def test_load_pco_ff_frame_background_subtracted(tmp_path):
    frames = np.ones((NF, H, W), dtype=np.float32) * 5.0
    frames[1, 2, 3] = 50.0  # a spike in frame 1
    _write_rocking(str(tmp_path), "rock__1", 0.0, 0.0, frames)
    h5p = os.path.join(str(tmp_path), "rock__1", "rock__1.h5")
    img = M.load_pco_ff_frame(h5p, "1.1/measurement/pco_ff", frame_index=1)
    assert img.shape == (H, W)
    assert img[2, 3] == pytest.approx(45.0)  # 50 - median(5) = 45
    # flat-background pixels: value - median == 0 (not negative -> not NaN)
    assert np.nansum(img) == pytest.approx(45.0)


# -- end to end ---------------------------------------------------------------
def test_run_saves_matched_layers(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    samy = [0.0, 0.001, 0.002]
    samz = [0.0, 0.001, 0.002]
    for i in range(3):
        _write_strain(str(raw), f"strain__{i + 1}", samy[i], samz[i])
        frames = np.random.default_rng(i).standard_normal((NF, H, W)) + 10.0
        _write_rocking(str(raw), f"rock__{i + 1}", samy[i], samz[i], frames)
    out = tmp_path / "matched_out"
    res = M.run(
        {
            "raw_root": str(raw),
            "strain_pattern": "strain__*",
            "rocking_pattern": "rock__*",
            "frame_index": 0,
            "match_threshold_mm": 0.001,
            "output_dir": str(out),
        }
    )
    assert res.n_strain == 3
    assert res.n_matched == 3 and res.n_saved == 3
    pngs = [p for p in os.listdir(res.layers_dir) if p.endswith(".png")]
    assert len(pngs) == 3


def test_run_skips_mismatched_frame_shape(tmp_path):
    """A matched rocking scan with a different detector shape is skipped, not fatal."""
    raw = tmp_path / "raw"
    raw.mkdir()
    samy = [0.0, 0.001, 0.002]
    samz = [0.0, 0.001, 0.002]
    for i in range(3):
        _write_strain(str(raw), f"strain__{i + 1}", samy[i], samz[i])
    rng = np.random.default_rng(0)
    _write_rocking(str(raw), "rock__1", samy[0], samz[0], rng.standard_normal((NF, H, W)) + 10)
    _write_rocking(str(raw), "rock__2", samy[1], samz[1], rng.standard_normal((NF, H, W)) + 10)
    # rock__3 has a wider detector -> cannot share the canvas built from rock__1
    _write_rocking(str(raw), "rock__3", samy[2], samz[2], rng.standard_normal((NF, H, W + 3)) + 10)
    res = M.run(
        {
            "raw_root": str(raw),
            "strain_pattern": "strain__*",
            "rocking_pattern": "rock__*",
            "match_threshold_mm": 0.001,
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert res.n_matched == 3 and res.n_saved == 2  # mismatched one skipped, no crash
    assert any("shape" in s for s in res.skipped)


def test_run_requires_raw_root():
    with pytest.raises(ValueError, match="raw_root"):
        M.run({"raw_root": ""})


# -- the median is blocked in-plane, and the frame must not move --------------


def _count_row_blocks(monkeypatch):
    """Record how many axis=1 blocks each `iter_blocks` traversal yielded.

    Empty if `load_pco_ff_frame` never blocks at all — which is what a reverted
    `ds[:]` looks like — so the precondition assertions cannot pass vacuously.
    """
    from dfxm.common import volumeio

    # Undo any earlier patch first, so a per-budget sweep wraps the pristine
    # function rather than the previous wrapper (which would leave every earlier
    # `seen` list still recording).
    monkeypatch.undo()
    real = volumeio.iter_blocks
    seen: list[int] = []

    def counting(dset, **kwargs):
        assert kwargs.get("axis") == 1, "matched must block in-plane, not along the frame axis"
        n = 0
        for item in real(dset, **kwargs):
            n += 1
            yield item
        seen.append(n)

    monkeypatch.setattr(volumeio, "iter_blocks", counting)
    return seen


def _write_stack(path, stack):
    with h5py.File(path, "w") as f:
        f.create_dataset("1.1/measurement/pco_ff", data=stack)
    return path


def _whole_stack_frame(stack, idx):
    """The pre-blocking form of `load_pco_ff_frame`, as the reference."""
    stack = stack.astype(np.float64)
    corrected = stack[idx] - np.nanmedian(stack, axis=0)
    corrected[corrected < 0] = np.nan
    return corrected


def test_matched_blocked_median_matches_whole_stack(tmp_path, monkeypatch):
    """A row-blocked median is the same median: it reduces along the frame axis."""
    rng = np.random.default_rng(9)
    stack = rng.normal(size=(11, 12, 14))
    stack[rng.random(stack.shape) < 0.05] = np.nan  # NaNs, so nanmedian is doing work
    path = _write_stack(str(tmp_path / "scan.h5"), stack)
    expected = _whole_stack_frame(stack, 3)
    assert np.isnan(expected).any() and np.isfinite(expected).mean() > 0.3, (
        "fixture must produce a frame that is neither all-NaN nor NaN-free"
    )

    seen = _count_row_blocks(monkeypatch)
    got = M.load_pco_ff_frame(path, "1.1/measurement/pco_ff", 3, budget_bytes=1024)

    assert seen and seen[0] >= 4, f"the budget must have split the rows, got {seen}"
    assert got.tobytes() == expected.tobytes(), "the blocked frame is not bit-equal"


def test_matched_frame_is_budget_independent(tmp_path, monkeypatch):
    """`axis=1` blocking is guaranteed by evidence, not construction — so check it.

    `volumeio`'s budget-independence is structural only for `axis=0` (a value's
    position in the flattened stream does not move with the block width). Under
    column/row blocking the global index does move, so every `axis=1` consumer
    owes a cross-budget check of its own. Same data, five budgets, compared as
    bytes.
    """
    rng = np.random.default_rng(17)
    stack = (rng.normal(size=(9, 40, 33)) * 250.0).astype(np.float32)
    stack[rng.random(stack.shape) < 0.07] = np.nan
    path = _write_stack(str(tmp_path / "scan.h5"), stack)

    digests = []
    block_counts = []
    for budget in (256, 2048, 16 * 1024, 256 * 1024, 64 << 20):
        seen = _count_row_blocks(monkeypatch)
        frame = M.load_pco_ff_frame(path, "1.1/measurement/pco_ff", 4, budget_bytes=budget)
        digests.append(frame.tobytes())
        block_counts.append(seen[0])

    assert len(set(block_counts)) >= 3, (
        f"the budgets must actually have changed the row blocking, got {block_counts}"
    )
    assert min(block_counts) == 1 and max(block_counts) == stack.shape[1], (
        f"the sweep must span one-block and one-row-per-block, got {block_counts}"
    )
    assert len(set(digests)) == 1, "the frame moved with the memory budget"
    assert digests[0] == _whole_stack_frame(stack, 4).tobytes(), (
        "every budget agrees, but not with the whole-stack median"
    )


def test_matched_two_dimensional_dataset_is_unchanged(tmp_path):
    """A 2-D pco_ff is returned as-is — no median, and NO negative clamp."""
    plane = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    path = _write_stack(str(tmp_path / "flat.h5"), plane)
    got = M.load_pco_ff_frame(path, "1.1/measurement/pco_ff", 0)
    assert np.array_equal(got, plane.astype(np.float64))
    assert not np.isnan(got).any(), "the 2-D path must not clamp negatives to NaN"


def test_matched_median_block_budget_never_buys_more_than_the_budget(tmp_path):
    """The conversion rounds DOWN: a block must never cost more working set than asked."""
    path = _write_stack(str(tmp_path / "scan.h5"), np.zeros((8, 16, 16), dtype=np.uint16))
    with h5py.File(path, "r") as f:
        ds = f["1.1/measurement/pco_ff"]
        itemsize = ds.dtype.itemsize
        for budget in (1 << 10, 1 << 16, 1 << 24):
            block_bytes = M._median_block_budget(ds, budget)
            working_set = block_bytes / itemsize * (itemsize + M.MEDIAN_WORKING_SET_PER_ELEMENT)
            assert working_set <= budget, (
                f"budget {budget} bought a {working_set:.0f} B working set"
            )


def test_matched_median_working_set_constant_covers_the_measured_cost(tmp_path):
    """`MEDIAN_WORKING_SET_PER_ELEMENT` must not sit below what the median costs.

    The budget test above recomputes the working set *from* the constant, so it
    holds for any value of it — 36, 8 or 1 — and cannot see the constant being
    wrong. The constant is a **divisor** in `_median_block_budget`: too small a
    value buys a block larger than was counted, and the blocks then quietly
    exceed the budget the whole conversion exists to respect. `rocking`'s
    sibling bound is measured; this is the same treatment for this one.

    Measures `load_pco_ff_frame` itself rather than a re-typed copy of its two
    lines, so the number tracks the shipped code.

    The sweep must span the **penalty region**. The cost is a ~35 B/element
    asymptote plus a per-block overhead the element count divides, so small
    frame counts cost the most: an earlier version of this test sampled only
    (11, 128, 128) and (41, 160, 160) — both sitting on the asymptote — and so
    passed against a constant of 36 that (3, 64, 64) exceeds by 1.4x. Frame
    counts 3/5/11/41 across small and large in-plane sizes, and the assertion is
    against the WORST case, never an average.
    """
    import gc
    import tracemalloc

    rng = np.random.default_rng(11)
    shapes = [
        (nf, n, n) for nf in (3, 5, 11, 41) for n in (64, 160)
    ]  # small nf = the expensive end
    measurements: dict[tuple[str, tuple[int, int, int]], float] = {}

    for dtype in ("uint16", "float32"):
        for shape in shapes:
            stack = rng.integers(0, 4000, size=shape).astype(dtype)
            path = _write_stack(str(tmp_path / f"ws_{dtype}_{shape[0]}_{shape[1]}.h5"), stack)
            itemsize = np.dtype(dtype).itemsize
            elems = int(np.prod(shape))

            with h5py.File(path, "r") as f:
                block_bytes = M._median_block_budget(f["1.1/measurement/pco_ff"], 1 << 30)
            assert block_bytes >= elems * itemsize, (
                "precondition: this budget must buy the whole stack as ONE block, so the "
                "measurement below is over a block of exactly `elems` elements"
            )

            del stack
            gc.collect()
            tracemalloc.start()
            base = tracemalloc.get_traced_memory()[0]
            frame = M.load_pco_ff_frame(path, "1.1/measurement/pco_ff", 1, budget_bytes=1 << 30)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

            # The constant prices what is held *besides* the block's own stored
            # bytes and the frame the call returns.
            measured = (peak - base - elems * itemsize - frame.nbytes) / elems
            assert measured > 8.0, (
                f"{dtype} {shape}: measured only {measured:.2f} B/element, at or below the "
                "bare `.astype(np.float64)` upcast — `np.nanmedian`'s own cost is not in "
                "this number, so the measurement is not seeing the thing it claims to bound"
            )
            measurements[(dtype, shape)] = measured

    worst_case, worst = max(measurements.items(), key=lambda kv: kv[1])

    # Precondition on the SWEEP, not on the code under test: at least one shape
    # must land in the penalty region — above the 36 B/element this constant used
    # to be — or the sweep has drifted back to sampling only the flat asymptote
    # and would pass whether or not the constant covers the real worst case.
    assert worst > 36.0, (
        f"precondition: the worst shape measured only {worst:.2f} B/element, so this "
        "sweep no longer reaches the small-frame-count penalty region it exists to "
        f"cover (measured: { {k: round(v, 2) for k, v in measurements.items()} })"
    )

    assert worst <= M.MEDIAN_WORKING_SET_PER_ELEMENT, (
        f"{worst_case[0]} {worst_case[1]}: one block costs {worst:.2f} B/element, over "
        f"the {M.MEDIAN_WORKING_SET_PER_ELEMENT} B MEDIAN_WORKING_SET_PER_ELEMENT charges "
        "it — `_median_block_budget` divides by that, so blocks now exceed the budget"
    )


def test_matched_peak_does_not_follow_the_stack_size(tmp_path):
    """The measured peak, not the model of it — a 40 MiB stack in a child process.

    Before the in-plane blocking this run peaked at 832 MiB and grew ~17.5 B per
    stored byte (measured: 20/40/80 MiB of stack -> 472/832/1526 MiB). It is now
    flat at ~250 MiB across the same sweep, so the limit below is a real
    discriminator rather than a rubber stamp.
    """
    from tests.peak_rss import assert_peak_under

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_strain(str(raw), "strain__1", 0.0, 0.0)
    folder = raw / "rock__1"
    folder.mkdir()
    frames = np.random.default_rng(3).integers(0, 4000, size=(80, 512, 512), dtype=np.uint16)
    with h5py.File(folder / "rock__1.h5", "w") as f:
        f.create_dataset("1.1/instrument/positioners/samy", data=0.0)
        f.create_dataset("1.1/instrument/positioners/samz", data=0.0)
        f.create_dataset("1.1/measurement/pco_ff", data=frames)
    assert frames.nbytes >= 40 * (1 << 20), "the stack must be large enough to show in RSS"
    del frames

    result = assert_peak_under(
        "dfxm.stages.matched:run",
        {
            "raw_root": str(raw),
            "strain_pattern": "strain__*",
            "rocking_pattern": "rock__*",
            "frame_index": 0,
            "match_threshold_mm": 0.001,
            "output_dir": str(tmp_path / "out"),
        },
        384 * (1 << 20),
    )
    assert result.n_saved == 1, "a peak measured on a run that produced nothing proves nothing"


def test_colormap_param_is_enum_dropdown():
    from dfxm.common.plotting import CMAP_CHOICES
    from dfxm.config.models import ParamType
    from dfxm.stages.matched import STAGE

    p = next(q for q in STAGE.params if q.name == "colormap")
    assert p.type is ParamType.ENUM
    assert tuple(p.choices) == CMAP_CHOICES
    assert p.default == "gray"


# -- the raw quantity group ---------------------------------------------------
def _one_layer_setup(tmp_path):
    """The smallest input run() will process: one matched strain/rocking pair."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_strain(str(raw), "strain__1", 0.0, 0.0)
    frames = np.random.default_rng(0).standard_normal((NF, H, W)) + 10.0
    _write_rocking(str(raw), "rock__1", 0.0, 0.0, frames)
    return {
        "raw_root": str(raw),
        "strain_pattern": "strain__*",
        "rocking_pattern": "rock__*",
        "frame_index": 0,
        "match_threshold_mm": 0.001,
        "output_dir": str(tmp_path / "out"),
        "colormap": "gray",
        "plot_style": asdict(PlotStyle(cmap_raw="turbo")),
    }


def _capture_layer_figure(monkeypatch):
    """Record the cmap/group handed to render.layer_figure, still rendering."""
    seen = {}
    real = M.Rnd.layer_figure

    def spy(layer, vmin, vmax, cmap, ex, ey, title, cbar, *, style=None, group=None):
        seen["cmap"] = cmap
        seen["group"] = group
        return real(layer, vmin, vmax, cmap, ex, ey, title, cbar, style=style, group=group)

    monkeypatch.setattr(M.Rnd, "layer_figure", spy)
    return seen


def test_run_renders_the_raw_quantity_group(tmp_path, monkeypatch):
    """matched draws rocking-curve frames — the "raw" group, like rocking.py."""
    params = _one_layer_setup(tmp_path)
    assert params["colormap"] == "gray"  # precondition: the fallback differs
    seen = _capture_layer_figure(monkeypatch)
    res = M.run(params)
    assert res.n_saved == 1  # precondition: a layer was actually drawn
    assert seen["group"] == "raw"
    assert seen["cmap"] == "turbo"  # the style's raw cmap, not the param


def test_the_export_rebuild_resolves_the_same_colormap_as_the_run(tmp_path, monkeypatch):
    """An exported figure must match the PNG the run saved."""
    params = _one_layer_setup(tmp_path)
    res = M.run(params)
    assert res.recorded and res.recorded[0].colormap == "gray"  # precondition
    specs = M.figures(res, params)
    assert specs  # precondition: there is something to rebuild
    seen = _capture_layer_figure(monkeypatch)
    specs[0].build(PlotStyle(cmap_raw="turbo"))
    assert seen["cmap"] == "turbo"
    assert seen["group"] == "raw"


def test_the_stage_colormap_is_still_the_fallback_without_a_group():
    from dfxm.common.plotting import resolve_cmap

    assert resolve_cmap(PlotStyle(cmap_raw="turbo"), None, fallback="magma") == "magma"


def test_a_headless_run_ignores_the_stage_colormap_too(tmp_path, monkeypatch):
    """No injected style is where a fallback could plausibly fire — it does not.

    `resolve_cmap` returns its fallback only for ``group=None``; a literal
    "raw" resolves against a bare ``PlotStyle()`` instead. So the stage's own
    dropdown colours nothing on any path, which is what the help text and
    docs/Usage.md now say.
    """
    from dfxm.common.plotting import style_from_params

    params = _one_layer_setup(tmp_path)
    params.pop("plot_style")
    params["colormap"] = "turbo"
    assert style_from_params(params) is None  # precondition: the headless path
    assert PlotStyle().cmap_raw == "gray"  # precondition: distinguishable from "turbo"
    seen = _capture_layer_figure(monkeypatch)
    res = M.run(params)
    assert res.n_saved == 1  # precondition: a layer was actually drawn
    assert seen["group"] == "raw"
    assert seen["cmap"] == "gray"  # the raw group's default, not the param
