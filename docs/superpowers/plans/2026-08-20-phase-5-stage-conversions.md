# Phase 5 — Stage Conversions to Bounded Memory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every volume stage completes the real STO2 dataset under an enforced memory ceiling, producing output bit-identical to an unconstrained run.

**Architecture:** Three families of fix. Two stages (`strain`, `mosaicity`) *build* volumes by collecting per-layer maps and `np.stack`ing them — they get an incremental HDF5 writer and never hold a volume at all. Three stages (`visualize`, `paraview`, `slices`) load whole volumes and run the fixed alignment chain — they get a lifecycle fix first, then a streaming alignment that blocks along Z, each block seeing the one row of context its interpolation reads. Three sites (`rocking`, `matched`, the two render loads) are each one localised change: a streaming quantile, in-plane blocking, and decimation on read. That last one covers **two** of the spec's thirteen sites — `gui/viewers.py:53` (the 3-D viewer) and `dfxm/viewer_jobs.py:20` (the rotation-video export child) — under one shared policy in `dfxm/common/volumeio.py`.

**Tech Stack:** Python 3, numpy, h5py, scipy (`interp1d`, `ndimage.shift`, `map_coordinates`), psutil, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-phase-5-stage-conversions-design.md` — read it before Task 1. Its parent is `docs/superpowers/specs/2026-08-20-machine-aware-robustness-design.md`.

## Global Constraints

- **`dfxm/` stays Qt-free.** Never import PySide6 or pyvista under `dfxm/`. Only Task 4 touches `gui/`.
- **One alignment.** `dfxm/common/alignment.py` holds the only implementation of the arithmetic. The streaming path calls the existing step functions; it does not reimplement them. The fixed order is `abs(FWHM) → ROI → samy X-shift → uniform-Z interp → centre` and is never reordered.
- **Budget-independence is the product guarantee.** For any `budget_bytes`, converted code produces bit-identical output. Verified with `tests/equivalence.py::assert_budget_independent`, which compares via `np.array_equal(..., equal_nan=True)` — NaN placement is part of the guarantee.
- **Never `np.sum`/`np.mean` across blocks.** numpy reduces pairwise with a 128-element base case, so summing whole and summing in blocks give different bits. Carry compensated state through `volumeio.neumaier_sum` instead.
- **`RunPlan.chunk_layers` is display-only.** Stages drive `volumeio` with `RunPlan.budget_bytes` alone.
- **Plotting:** build figures with the explicit `matplotlib.figure.Figure` API. Never `pyplot`, never `matplotlib.use(...)`.
- **User-facing input errors** raise `StageUserError(message, hint)`.
- **Docs contract:** every task that changes a stage's behaviour, parameters, inputs/outputs, or public functions updates `docs/Usage.md` (user-visible behaviour) **and** `docs/Codebase.md` (code structure) in the same commit. This is not a follow-up.
- **Suite command on this box:** `python3 -m pytest -q --deselect tests/test_gui_viewer3d.py` (in-process Qt GL segfaults). GUI smoke is `DISPLAY= python3 -u tests/gui_smoke.py`, and step `[41]` is intermittently GL-flaky — retry once.
- **Lint:** `ruff check . && ruff format .` must be clean. `ruff format` runs automatically on Write/Edit via the settings hook.
- **Read before first Edit.** Any file not created in your session must be Read once before its first Edit. Never reconstruct an `old_string` from memory — `hint=` strings contain em-dashes and sit at 12 or 16 spaces depending on nesting.

## File Structure

**Created:**

- `tests/peak_rss.py` — peak-RSS measurement harness (not a pytest file; no `test_` prefix, same convention as `tests/equivalence.py`).

**Modified:**

- `dfxm/common/h5io.py` — gains `StackedVolumeFile`, the incremental layer writer shared by `strain` and `mosaicity`.
- `dfxm/common/volumeio.py` — gains `axis=1` on `iter_blocks`, plus `iter_with_context`, `dataset_blocks`, `stream_mean`, `stream_minmax`, `stream_quantile`.
- `dfxm/common/alignment.py` — gains a `pad` parameter on `apply_samy_shifts_to_volume`, a `z_uniform` parameter on `interpolate_to_uniform_z`, and `StreamedAlignment` + `align_volume_streamed`. `align_volume` becomes the in-core façade.
- `dfxm/runner.py` — gains a public `pid` property so the harness can watch the child.
- `dfxm/stages/strain.py`, `mosaicity.py` — incremental writes; `save_stacked_volume` removed.
- `dfxm/stages/visualize.py`, `paraview.py` — per-field loading, then streamed alignment.
- `dfxm/stages/slices.py` — lifecycle fix, then Z-blocked gather.
- `dfxm/stages/rocking.py`, `matched.py` — streaming quantile; in-plane blocked median.
- `gui/viewers.py` — decimation on read.
- `docs/Usage.md`, `docs/Codebase.md` — per the docs contract, in the same commit as each change.

---

## Wave 1 — Lifecycle fixes

No new machinery, no numeric change. Every task in this wave must leave output **byte-identical** to before, which makes each one cheap to verify and cheap to reject.

### Task 1: Incremental layer writes for `strain` and `mosaicity`

Both stages accumulate every layer in RAM and then `np.stack`, so their peak is two-to-five whole volumes for a product that is written once and never re-read. One shared writer fixes both.

**Files:**
- Modify: `dfxm/common/h5io.py` (append `StackedVolumeFile` after `iter_dataset_sizes`)
- Modify: `dfxm/stages/strain.py:791-804` (remove `save_stacked_volume`), `dfxm/stages/strain.py:874-925` (`run`)
- Modify: `dfxm/stages/mosaicity.py:491-538` (`run`)
- Test: `tests/test_common_h5io.py`, `tests/test_stage_strain.py`, `tests/test_stage_mosaicity.py`
- Docs: `docs/Codebase.md` (new `StackedVolumeFile`; `save_stacked_volume` removed)

**Interfaces:**
- Produces: `dfxm.common.h5io.StackedVolumeFile(path, *, compression="gzip")` with methods `append(dataset_path: str, layer: np.ndarray) -> None`, `shape(dataset_path: str) -> tuple[int, ...]`, `set_attrs(**attrs) -> None`, `close() -> None`, `abort() -> None`, and context-manager support (commit on clean exit, abort on exception).
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test for the writer**

Add to `tests/test_common_h5io.py`:

```python
import h5py
import numpy as np
import pytest

from dfxm.common.h5io import StackedVolumeFile


def test_stacked_volume_file_appends_layers(tmp_path):
    path = str(tmp_path / "stacked.h5")
    layers = [np.full((3, 4), i, dtype=np.float64) for i in range(5)]
    with StackedVolumeFile(path, compression=None) as out:
        for layer in layers:
            out.append("strain", layer)
        out.set_attrs(num_layers=len(layers))
        assert out.shape("strain") == (5, 3, 4)
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["strain"][:], np.stack(layers, axis=0))
        assert f.attrs["num_layers"] == 5


def test_stacked_volume_file_rejects_shape_change(tmp_path):
    path = str(tmp_path / "stacked.h5")
    with pytest.raises(ValueError, match="differing shapes"):
        with StackedVolumeFile(path, compression=None) as out:
            out.append("strain", np.zeros((3, 4)))
            out.append("strain", np.zeros((3, 5)))


def test_stacked_volume_file_leaves_no_file_on_failure(tmp_path):
    path = str(tmp_path / "stacked.h5")
    with pytest.raises(ValueError):
        with StackedVolumeFile(path, compression=None) as out:
            out.append("strain", np.zeros((3, 4)))
            raise ValueError("boom")
    assert not (tmp_path / "stacked.h5").exists()
    assert list(tmp_path.glob("*.part")) == []


def test_stacked_volume_file_nested_group_paths(tmp_path):
    path = str(tmp_path / "stacked.h5")
    with StackedVolumeFile(path, compression=None) as out:
        out.append("chi/Center_of_mass", np.zeros((2, 2)))
        out.append("mu/FWHM", np.zeros((2, 2)))
        out.append("chi/Center_of_mass", np.ones((2, 2)))
    with h5py.File(path, "r") as f:
        assert f["chi/Center_of_mass"].shape == (2, 2, 2)
        assert f["mu/FWHM"].shape == (1, 2, 2)
```

The last test matters: `mosaicity` writes into `chi/` and `mu/` groups, and a layer may be missing one dataset, so per-dataset layer counts legitimately differ.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_common_h5io.py -k stacked_volume_file -v`
Expected: FAIL — `ImportError: cannot import name 'StackedVolumeFile'`

- [ ] **Step 3: Implement `StackedVolumeFile`**

Append to `dfxm/common/h5io.py`:

```python
class StackedVolumeFile:
    """Build a (Z, Y, X) volume file one layer at a time.

    ``strain`` and ``mosaicity`` used to collect every layer in a list and
    ``np.stack`` it, which costs two whole volumes for a product that is
    written once and never re-read. Appending into a resizable dataset costs
    one layer.

    Writes to ``<path>.part`` and renames on a clean close, so a failure
    mid-run leaves nothing behind — the same all-or-nothing behaviour the
    write-at-the-end version had for free.
    """

    def __init__(self, path: str, *, compression: str | None = "gzip") -> None:
        self._path = path
        self._part = path + ".part"
        self._compression = compression
        self._shapes: dict[str, set[tuple[int, ...]]] = {}
        self._f = h5py.File(self._part, "w")

    def append(self, dataset_path: str, layer: np.ndarray) -> None:
        """Add one 2-D layer to *dataset_path*, creating the dataset if needed."""
        layer = np.asarray(layer)
        seen = self._shapes.setdefault(dataset_path, set())
        seen.add(tuple(layer.shape))
        if dataset_path not in self._f:
            kw: dict = {}
            if self._compression:
                kw["compression"] = self._compression
                if self._compression == "gzip":
                    kw["compression_opts"] = 4
            self._f.create_dataset(
                dataset_path,
                shape=(0, *layer.shape),
                maxshape=(None, *layer.shape),
                dtype=layer.dtype,
                chunks=(1, *layer.shape),
                **kw,
            )
        dset = self._f[dataset_path]
        if tuple(layer.shape) != tuple(dset.shape[1:]):
            raise ValueError(f"{dataset_path}: maps have differing shapes {seen}; fix ROI")
        n = dset.shape[0]
        dset.resize(n + 1, axis=0)
        dset[n] = layer

    def shape(self, dataset_path: str) -> tuple[int, ...]:
        return tuple(int(d) for d in self._f[dataset_path].shape)

    def datasets(self) -> list[str]:
        return sorted(self._shapes)

    def set_attrs(self, **attrs) -> None:
        for key, value in attrs.items():
            self._f.attrs[key] = value

    def close(self) -> None:
        """Flush, close and move the part file into place."""
        self._f.close()
        os.replace(self._part, self._path)

    def abort(self) -> None:
        """Close and discard; never masks the caller's exception."""
        try:
            self._f.close()
        except Exception:  # noqa: BLE001 - already-broken file
            pass
        try:
            os.unlink(self._part)
        except OSError:
            pass

    def __enter__(self) -> "StackedVolumeFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False
```

`h5io.py` already imports `os`, `h5py` and `numpy as np`; confirm before adding imports.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_common_h5io.py -k stacked_volume_file -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Convert `strain.run` to the writer**

In `dfxm/stages/strain.py`, delete `save_stacked_volume` (lines 791-804) entirely, and rewrite the accumulation in `run`. Replace `slices: list[np.ndarray] = []` with nothing (keep `names`), and restructure so the writer wraps the loop:

```python
    out_dir = p["output_dir"] or os.path.join(default_out_root, "strain_maps")
    result = StrainResult(output_dir=out_dir)

    stacked_path = os.path.join(default_out_root, p["stacked_filename"])
    names: list[str] = []
    with StackedVolumeFile(stacked_path, compression="gzip") as out:
        for i, (name, maps_path) in enumerate(work):
            progress(i / len(work), f"strain: {name}")
            if not os.path.exists(maps_path):
                result.skipped.append(f"{name}: {maps_filename} not found")
                continue
            try:
                strain, layer = process_maps_file(
                    maps_path,
                    name,
                    ccmth_com_path=p["ccmth_com_path"],
                    ccmth_ref_deg=float(p["ccmth_ref_deg"]),
                    pixel_size_x_um=float(p["pixel_size_x_um"]),
                    pixel_size_y_um=float(p["pixel_size_y_um"]),
                    roi=roi,
                    vlim=vlim,
                    out_dir=out_dir,
                    save_plots=bool(p["save_plots"]),
                    style=style,
                )
            except StageUserError:
                # Out-of-bounds ROI etc. is an input problem affecting every layer the
                # same way — stop the run with a clear message rather than skip-and-continue.
                raise
            except (KeyError, OSError, ValueError) as exc:
                result.skipped.append(f"{name}: {exc}")
                continue
            out.append("strain", strain)
            del strain
            names.append(name)
            result.layers.append(layer)

        if not names:
            out.abort()
            progress(1.0, "no strain layers produced")
            return result

        out.set_attrs(
            num_layers=len(names),
            source_folders="\n".join(names),
            description="Stacked 3D strain volume (cot, ccmth-only)",
            ccmth_ref_deg=float(p["ccmth_ref_deg"]),
            scale_x_um=float(p["pixel_size_x_um"]),
            scale_y_um=float(p["pixel_size_y_um"]),
        )
        shape = out.shape("strain")

    result.stacked_path = stacked_path
    result.volume_shape = shape
    progress(1.0, f"stacked {len(names)} layers -> {os.path.basename(stacked_path)}")
    return result
```

Three things to get right. The `del strain` matters — without it the layer stays alive across the next `process_maps_file` call. The early return calls `out.abort()` *and* returns inside the `with`, so `__exit__` sees no exception and would otherwise commit an empty file; `abort()` before returning makes the second close a no-op on an already-closed file, which `abort` tolerates. And the shape-mismatch check that used to live after the loop (`shapes = {s.shape for s in slices}`) is now `StackedVolumeFile.append`'s `ValueError`, which keeps the "differing shapes … fix ROI" wording that `tests/test_stage_strain.py` may assert on — grep for `differing shapes` before you run.

Add the import at the top of `strain.py`: `from dfxm.common.h5io import StackedVolumeFile` (follow the existing import grouping).

- [ ] **Step 6: Convert `mosaicity.run` to the writer**

In `dfxm/stages/mosaicity.py`, replace the `collected` dict and the write block (lines 491-538):

```python
    compression = None if p["compression"] == "none" else p["compression"]
    stacked_path = os.path.join(default_out_root, p["stacked_filename"])
    result = MosaicityResult()

    with StackedVolumeFile(stacked_path, compression=compression) as out:
        for i, (name, maps_path) in enumerate(work):
            progress(i / len(work), f"mosaicity: {name}")
            if not os.path.exists(maps_path):
                result.skipped.append(f"{name}: {maps_filename} not found")
                continue
            try:
                with h5py.File(maps_path, "r") as f:
                    data = {key: _read_dataset(f, path) for key, path in config.items()}
            except OSError as exc:
                result.skipped.append(f"{name}: {exc}")
                continue
            if all(v is None for v in data.values()):
                result.skipped.append(f"{name}: no datasets")
                continue
            for key, arr in data.items():
                if arr is not None:
                    group_name, ds_name = routing[key]
                    out.append(f"{group_name}/{ds_name}", arr)
            del data
            result.layers.append(name)

        if not result.layers:
            out.abort()
            progress(1.0, "no mosaicity layers produced")
            return result

        for ds_path in out.datasets():
            result.datasets[f"/{ds_path}"] = out.shape(ds_path)
        out.set_attrs(
            num_layers=len(result.layers),
            source_folders="\n".join(result.layers),
            description="Stacked 3D volumes from 2D darfix maps",
        )

    result.stacked_path = stacked_path
    progress(1.0, f"stacked {len(result.layers)} layers -> {os.path.basename(stacked_path)}")
    return result
```

`_read_dataset` still reads one layer's 2-D map, which is small; only the accumulation changed. Note that `result.datasets` keys keep their leading slash (`/chi/Center_of_mass`) to match the previous format — `tests/test_stage_mosaicity.py` asserts on those keys.

The `groups = {g: f.require_group(g) for ...}` line is gone: `StackedVolumeFile.append` creates intermediate groups implicitly through the `"chi/Center_of_mass"` path. If a downstream consumer requires an *empty* group to exist for an absent dataset, that behaviour changes — grep `require_group` in `dfxm/` and `gui/` and confirm nothing depends on it before deleting the line.

Add `from dfxm.common.h5io import StackedVolumeFile` to the imports.

- [ ] **Step 7: Add byte-identity tests for both stages**

The point of Wave 1 is that nothing about the *product* changes. Add to `tests/test_stage_strain.py`:

```python
def test_strain_volume_matches_layerwise_stack(tmp_path):
    """The written volume equals np.stack of the per-layer maps, as before."""
    from dfxm.stages import strain

    params = _minimal_strain_params(tmp_path)  # reuse this module's existing fixture helper
    result = strain.run(params)
    with h5py.File(result.stacked_path, "r") as f:
        volume = f["strain"][:]
        assert f.attrs["num_layers"] == len(result.layers)
    expected = np.stack(
        [strain.process_maps_file(lr.maps_path, lr.name, **_process_kwargs(params))[0]
         for lr in result.layers],
        axis=0,
    )
    assert np.array_equal(volume, expected, equal_nan=True)
```

If `tests/test_stage_strain.py` has no `_minimal_strain_params`/`_process_kwargs` helper, read the module's existing synthetic-input fixture and use whatever it provides — do not invent a new fixture format. Add the equivalent test to `tests/test_stage_mosaicity.py` for one of the four datasets.

- [ ] **Step 8: Run the affected suites**

Run: `python3 -m pytest tests/test_common_h5io.py tests/test_stage_strain.py tests/test_stage_mosaicity.py tests/test_stage_estimates.py -q`
Expected: PASS. `test_stage_estimates.py` is in the list because both estimators' docstrings describe the old peak model — they are *not* corrected here (Task 13 owns that), but the tests must still pass.

- [ ] **Step 9: Update the docs**

`docs/Codebase.md`: add `StackedVolumeFile` under the `dfxm/common/h5io.py` section; remove `save_stacked_volume` from the `strain.py` section. `docs/Usage.md`: no user-visible change — the products are identical — so it needs no edit for this task. State that explicitly in the commit body rather than silently skipping it.

- [ ] **Step 10: Commit**

```bash
ruff check . && ruff format .
git add dfxm/common/h5io.py dfxm/stages/strain.py dfxm/stages/mosaicity.py tests/ docs/Codebase.md
git commit -m "perf: write strain and mosaicity volumes layer-by-layer

Both stages collected every layer in RAM and then np.stack'd, so their peak
was two (strain) to five (mosaicity) whole volumes for a product written once
and never re-read. StackedVolumeFile appends into a resizable dataset, so the
peak is one layer.

Products are unchanged; what differs is the HDF5 layout parameters — maxshape
is now unlimited on the layer axis, and each volume is chunked one layer per
chunk (under the default gzip the old code was already chunked, just with
h5py's auto-chosen chunk shape; only mosaicity at compression=none was
genuinely contiguous).
Usage.md needs no edit: nothing user-visible changed."
```

---

### Task 2: Per-field loading in `visualize` and `paraview`

Both stages load *every* field into a dict before touching any of them. In `visualize` that dict is never freed, so it stays alive through the strain section too — which is why its estimated peak sums both files' raw bytes instead of maxing them.

**Files:**
- Modify: `dfxm/stages/visualize.py:491-503` (`load_mosa_datasets`), `:656-691` (`run`), `:786-791` (`aligned_field`), `:929-932` (`figures` closure)
- Modify: `dfxm/stages/paraview.py:637-649` (`load_mosa_datasets`), `:687-712` (`_process_mosaicity`)
- Test: `tests/test_stage_visualize.py`, `tests/test_stage_paraview.py`
- Docs: `docs/Codebase.md`

**Interfaces:**
- Produces: in **both** `visualize.py` and `paraview.py` (each keeps its own copy, as today), `mosa_field_names(path: str) -> list[str]` and `load_mosa_field(path: str, name: str) -> np.ndarray | None`. Field names keep the existing `f"{group}_{ds.replace(' ', '_')}"` convention, so every downstream `_display_info(name)` call is unaffected.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_visualize.py`:

```python
def test_mosa_field_names_matches_dict_keys(tmp_path):
    """The lazy API enumerates exactly what the old eager dict contained."""
    from dfxm.stages import visualize

    path = _write_mosa_volume(tmp_path)  # reuse this module's existing fixture helper
    names = visualize.mosa_field_names(path)
    assert names == sorted(names), "names must be deterministic across runs"
    for name in names:
        field = visualize.load_mosa_field(path, name)
        assert field is not None and field.ndim == 3


def test_load_mosa_field_unknown_name_returns_none(tmp_path):
    from dfxm.stages import visualize

    path = _write_mosa_volume(tmp_path)
    assert visualize.load_mosa_field(path, "not_a_field") is None
```

The `sorted` assertion is deliberate: `load_mosa_datasets` iterated `f[group].keys()`, whose order h5py does not guarantee across files, and the run's field order feeds `result.datasets` ordering. Sorting makes it explicit rather than incidental.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_stage_visualize.py -k mosa_field -v`
Expected: FAIL — `AttributeError: module 'dfxm.stages.visualize' has no attribute 'mosa_field_names'`

- [ ] **Step 3: Replace the eager loader in `visualize.py`**

```python
def mosa_field_names(path) -> list[str]:
    """Field names in a mosaicity volume file, without reading any data."""
    names = []
    with h5py.File(path, "r") as f:
        for group in ("chi", "mu"):
            if group in f:
                for ds in f[group].keys():
                    names.append(f"{group}_{ds.replace(' ', '_')}")
    return sorted(names)


def load_mosa_field(path, name):
    """One field from a mosaicity volume file, or None if absent."""
    with h5py.File(path, "r") as f:
        for group in ("chi", "mu"):
            if group not in f:
                continue
            for ds in f[group].keys():
                if f"{group}_{ds.replace(' ', '_')}" == name:
                    return f[group][ds][:]
    return None
```

Delete `load_mosa_datasets`.

- [ ] **Step 4: Convert `visualize.run`'s mosaicity loop**

Replace lines 656-658 and the loop body's use of `raw`:

```python
        names = mosa_field_names(mosa_file)
        samy, samz = _read_motors(raw_root, p["mosa_pattern"], p["samy_path"], p["samz_path"])
        for i, name in enumerate(names):
            progress(0.1 + 0.4 * i / max(1, len(names)), f"mosaicity: {name}")
            raw = load_mosa_field(mosa_file, name)
            if raw is None:
                continue
            title, cbar, group = _display_info(name)
            cmap = resolve_cmap(style, group)
            data, z_pos, scale_z = _align(
                raw, samy, samz, scale_x=scale_x, samy_direction=samy_dir, roi_x=roi_x, roi_y=roi_y
            )
            del raw
            ...  # the rest of the body is unchanged, but every `len(datasets)` becomes `len(names)`
```

The `del raw` is the point of the task: `_align` returns a new array, so the source field is dead from that line on. Without the `del`, `raw` stays bound until the next iteration rebinds it — which is exactly the bug being fixed, just moved.

Also update the two other call sites: `aligned_field` (line ~786) becomes

```python
        raw = load_mosa_field(p["mosa_volume_file"], name)
        if raw is None:
            raise KeyError(f"field {name!r} not in {p['mosa_volume_file']!r}")
        data, _z, scale_z = _align(raw, ...)
```

and the `figures` closure (line ~929) likewise. Both currently load every field to use one.

- [ ] **Step 5: Convert `paraview._process_mosaicity`**

`paraview` cannot go fully per-field yet: `save_volumes_as_pvti` takes a dict of *all* fields because a `.vti` piece carries every field. What it can do now is stop holding the **raw** dict — only the aligned ones need to survive the loop.

```python
def _process_mosaicity(p, out_dir, scale_x, scale_y, samy_dir, roi_x, roi_y) -> ExportInfo | None:
    names = mosa_field_names(p["mosa_volume_file"])
    if not names:
        return None
    samy, samz = _motors(p["raw_root"], p["mosa_pattern"], p["samy_path"], p["samz_path"])

    processed = {}
    scale_z = None
    z_positions = None
    for name in names:
        raw = load_mosa_field(p["mosa_volume_file"], name)
        if raw is None:
            continue
        is_com = "Center_of_mass" in name
        is_fwhm = "FWHM" in name
        if is_fwhm and bool(p["abs_mosa_fwhm"]):
            raw = np.abs(raw)
        data = A.apply_roi_3d(raw, roi_x, roi_y)
        if len(samy) > 0:
            data = A.apply_samy_shifts_to_volume(data, samy, scale_x, samy_dir)
        del raw
        if len(samz) > 0:
            data, z_pos, sz = A.interpolate_to_uniform_z(data, samz)
        else:
            sz = 2.0
            z_pos = np.arange(data.shape[0], dtype=float) * sz
        if scale_z is None:
            scale_z, z_positions = sz, z_pos
        if is_com and bool(p["center_mosa_com"]):
            data, _ = A.center_around_zero(data, p["center_method"])
        processed[name] = data
    ...  # unchanged from the `origin = (` line onward
```

The `del raw` placement is exact and load-bearing: `apply_roi_3d` returns a **view**, so `data` keeps `raw` alive until `apply_samy_shifts_to_volume` allocates a new array. Deleting earlier would be wrong; deleting later wastes the saving. When `samy` is empty the shift is skipped and `data` remains a view of `raw`, so the `del` frees nothing — correct, and harmless.

Add the same `mosa_field_names`/`load_mosa_field` pair to `paraview.py` and delete its `load_mosa_datasets`.

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest tests/test_stage_visualize.py tests/test_stage_paraview.py tests/test_gui_viewers.py -q`
Expected: PASS. `test_gui_viewers.py` is included because `gui/viewers.py::_visualize_load` calls `visualize.aligned_field`.

Then grep for stragglers: `grep -rn "load_mosa_datasets" --include=*.py .` must return nothing.

- [ ] **Step 7: Update the docs and commit**

`docs/Codebase.md`: replace `load_mosa_datasets` with `mosa_field_names` + `load_mosa_field` in both the `visualize.py` and `paraview.py` sections. `docs/Usage.md`: no user-visible change.

```bash
ruff check . && ruff format .
git add dfxm/stages/visualize.py dfxm/stages/paraview.py tests/ docs/Codebase.md
git commit -m "perf: load mosaicity fields one at a time in visualize and paraview

load_mosa_datasets read every field into a dict before any of them was used.
In visualize that dict was never freed, so it stayed alive through the strain
section too. Fields are now enumerated by name and read individually, and the
raw array is dropped as soon as alignment has copied out of it.

paraview still holds all *aligned* fields, because a .vti piece carries every
field; that goes away when the piece writer starts streaming."
```

---

### Task 3: Free the previous prepared volume in `slices`

`prep` is rebound each iteration, so the previous volume stays fully alive while the next is being built — the estimator's "pairs the current volume's own load peak with the largest OTHER selected volume's footprint" term is exactly this.

**Files:**
- Modify: `dfxm/stages/slices.py:731-750` (`prepare_volume`), `:1379-1385` (the volumes loop)
- Test: `tests/test_stage_slices.py`
- Docs: none (no interface or behaviour change)

**Interfaces:**
- Consumes / Produces: nothing. This task changes only object lifetimes.

- [ ] **Step 1: Free `raw` inside `prepare_volume`**

In the `source == "stacked"` branch, `raw` is a float64 read that stays alive behind the `apply_roi_3d` view until the shift allocates:

```python
        samy, samz, _ = _motors(cfg, p)
        data = A.apply_roi_3d(raw, cfg.get("roi_x"), cfg.get("roi_y"))
        if len(samy) > 0:
            data = A.apply_samy_shifts_to_volume(data, samy, scale_x, samy_dir)
            del raw
        if len(samz) > 0:
            data, _z, scale_z = A.interpolate_to_uniform_z(data, samz)
```

The `del` sits **inside** the `if`, because when the shift is skipped `data` is still a view of `raw` and deleting the name would not free it anyway — but putting the `del` outside the `if` would be a latent bug the moment someone made the branch unconditional.

- [ ] **Step 2: Free the previous `prep` in the loop**

```python
        prep = None
        for vi, cfg in enumerate(volumes):
            progress(0.1 + 0.85 * vi / len(volumes), f"slicing {cfg['kind']} {cfg['dataset_path']}")
            prep = None  # release the previous volume before building the next
            try:
                prep = prepare_volume(cfg, p, scale_x, scale_y, samy_dir, style=style)
            except (KeyError, OSError, ValueError) as exc:
                result.skipped.append(f"{cfg['dataset_path']}: {exc}")
                continue
```

- [ ] **Step 3: Prove the lifetime actually changed**

A test that asserts a `del` happened is usually a test of nothing. This one can be real, because the object is observable through a weak reference:

```python
def test_slices_releases_previous_volume(tmp_path, monkeypatch):
    """The previous prepared volume is dead before the next one is built."""
    import weakref

    from dfxm.stages import slices

    seen: list = []
    real = slices.prepare_volume

    def spy(cfg, p, *args, **kwargs):
        # Every previously prepared volume must already be collectable.
        assert all(ref() is None for ref in seen), "previous volume still alive"
        prep = real(cfg, p, *args, **kwargs)
        seen.append(weakref.ref(prep["data"]))
        return prep

    monkeypatch.setattr(slices, "prepare_volume", spy)
    params = _two_volume_slices_params(tmp_path)  # reuse this module's fixture helper
    slices.run(params)
    assert len(seen) >= 2, "test needs at least two volumes to be meaningful"
```

If `tests/test_stage_slices.py` has no two-volume fixture, build one from the existing single-volume helper by pointing both `mosa_volume_file` and `strain_volume_file` at synthetic files. The `len(seen) >= 2` assertion guards against the test silently becoming vacuous.

Note this test will fail until Step 2 is applied — write it, watch it fail, then apply. If it passes *before* Step 2, the fixture only produces one volume and the test is not yet meaningful.

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_stage_slices.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check . && ruff format .
git add dfxm/stages/slices.py tests/test_stage_slices.py
git commit -m "perf: release the previous prepared volume before building the next

slices rebound `prep` each iteration, so the previous volume stayed fully
alive while the next was being prepared — the peak paired the current
volume's load with the largest other selected volume. Also frees the raw
float64 read once the samy shift has copied out of it."
```

---

### Task 4: Decimate the 3-D viewer's volume on read

Two of the spec's thirteen sites, one conversion:

- `gui/viewers.py:53` (the on-screen 3-D viewer) reads the whole volume, upcasts it with `.astype(float)`, and then builds `vol[np.isfinite(vol)]` for the percentile clim — a second full-size copy.
- `dfxm/viewer_jobs.py:20` (`_load_volume`, the rotation-video export child) reads the same dataset with a plain `[:]`.

VTK needs the array whole, so streaming cannot help; decimation can. This is a display path, so coarsening is allowed, but it must be visible. Both loads must decimate by the **same policy**, or the exported video and the view it came from drift apart — so the policy (`display_headroom_bytes`, `display_decimation`, `decimation_note`) lives in `dfxm/common/volumeio.py`, which both may import (`dfxm/` may not import `gui/`). Each still measures headroom in its own process, so the *factor* can differ; the note names the one actually used.

**Files:**
- Modify: `gui/viewers.py:25-74` (`LoadedVolume`, `_rocking_source`)
- Modify: `dfxm/viewer_jobs.py:17-46` (`_load_volume`), `dfxm/common/volumeio.py` (the shared display policy)
- Test: `tests/test_gui_viewers.py`, `tests/test_viewer_jobs.py`, `tests/test_common_volumeio.py`
- Docs: `docs/Usage.md` (user-visible coarsening), `docs/Codebase.md`

**Interfaces:**
- Produces: `LoadedVolume` gains `decimation: int = 1` and `notes: tuple[str, ...] = ()`. Consumers that construct `LoadedVolume` positionally must be checked — `grep -rn "LoadedVolume(" --include=*.py .` before editing.

- [ ] **Step 1: Write the failing test**

```python
def test_rocking_source_decimates_when_over_headroom(tmp_path, monkeypatch):
    import numpy as np
    import h5py
    from gui import viewers

    path = str(tmp_path / "aligned.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=np.random.rand(16, 16, 16))
        f.attrs["scale_x_um_per_px"] = 1.0

    monkeypatch.setattr(viewers, "_viewer_headroom_bytes", lambda: 1024)
    loaded = viewers._rocking_source(path, "sum_intensity")()
    assert loaded.decimation > 1
    assert loaded.volume.shape[0] < 16
    assert any("decimat" in n.lower() for n in loaded.notes)


def test_rocking_source_full_fidelity_when_it_fits(tmp_path, monkeypatch):
    import numpy as np
    import h5py
    from gui import viewers

    path = str(tmp_path / "aligned.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=np.random.rand(8, 8, 8))

    monkeypatch.setattr(viewers, "_viewer_headroom_bytes", lambda: 1 << 30)
    loaded = viewers._rocking_source(path, "sum_intensity")()
    assert loaded.decimation == 1
    assert loaded.volume.shape == (8, 8, 8)
    assert loaded.notes == ()
```

The second test is the fast-path guard: a machine with room must not pay for a small machine's safety.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_gui_viewers.py -k rocking_source -v`
Expected: FAIL — `AttributeError: module 'gui.viewers' has no attribute '_viewer_headroom_bytes'`

- [ ] **Step 3: Implement**

```python
@dataclass
class LoadedVolume:
    """One loaded, ready-to-render 3-D volume plus its display metadata."""

    volume: "np.ndarray"
    spacing: tuple
    cmap: str
    clim: tuple | None
    cbar_label: str
    group: str | None
    decimation: int = 1
    notes: tuple[str, ...] = ()


def _viewer_headroom_bytes() -> int:
    """How much RAM a viewer load may use. Wrapped so tests can shrink it."""
    from dfxm.common import advice, machine

    return advice.headroom_bytes(machine.profile())


def _decimation_for(dset) -> int:
    """Smallest power-of-two stride bringing *dset* within the viewer's headroom."""
    from dfxm.common.volumeio import volume_bytes

    budget = _viewer_headroom_bytes()
    # float64 is what the viewer holds, whatever the stored dtype.
    needed = volume_bytes(dset) // max(1, dset.dtype.itemsize) * 8
    step = 1
    while step < 16 and needed // (step**3) > budget:
        step *= 2
    return step


def _rocking_source(aligned_path: str, dataset: str) -> Callable[[], LoadedVolume]:
    def _load() -> LoadedVolume:
        notes: list[str] = []
        with h5py.File(aligned_path, "r") as f:
            dset = f[dataset]
            full_shape = tuple(int(d) for d in dset.shape)
            step = _decimation_for(dset)
            vol = dset[::step, ::step, ::step].astype(float)
            sx = float(f.attrs.get("scale_x_um_per_px", 1.0))
            sy = float(f.attrs.get("scale_y_um_per_px", 1.0))
            sz = float(f.attrs.get("scale_z_um_per_px", 1.0))
        if step > 1:
            notes.append(
                f"decimated {step}x for display ({full_shape[2]}x{full_shape[1]}x{full_shape[0]} "
                f"exceeds this machine's memory headroom) — the stored data is unchanged"
            )
        finite = vol[np.isfinite(vol)]
        clim = (
            (float(np.percentile(finite, 1)), float(np.percentile(finite, 99)))
            if finite.size
            else None
        )
        del finite
        from dfxm.common.plotting import resolve_cmap

        return LoadedVolume(
            volume=vol,
            spacing=(sx * step, sy * step, sz * step),
            cmap=resolve_cmap(None, "raw"),
            clim=clim,
            cbar_label="Intensity",
            group="raw",
            decimation=step,
            notes=tuple(notes),
        )

    return _load
```

Two details that are easy to get wrong. **Spacing must be multiplied by the stride** — a decimated volume with unscaled spacing renders at the wrong physical size, which is a silently wrong picture rather than a visibly coarse one. And `finite` is still a full-size copy of the *decimated* array; the `del` bounds it rather than eliminating it, which is honest — eliminating it would need a streaming percentile, and by this point the array already fits.

- [ ] **Step 4: Surface the note in the viewer**

Find where `LoadedVolume` reaches the 3-D window (`grep -rn "LoadedVolume" gui/`) and render `notes` wherever that window already shows text — `gui/widgets/viewer3d_window.py` is the likely home. Follow the existing pattern for messages in that window rather than adding a new mechanism. If the window has no text surface at all, add the note to its title bar; do not invent a banner.

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_gui_viewers.py tests/test_viewer_jobs.py -q`
Expected: PASS

Then the smoke test, since this is GUI code: `DISPLAY= python3 -u tests/gui_smoke.py` — steps `[1]`-`[40]` must pass; `[41]` is the GL step and is intermittently flaky, so retry it once before treating a failure as real.

- [ ] **Step 6: Update docs and commit**

`docs/Usage.md`: in the 3-D viewer section, state that a volume too large for the machine's memory is decimated for display, that the factor is shown, and that the stored data is untouched. This is the one user-visible coarsening in the whole phase, so it must be documented plainly. `docs/Codebase.md`: `LoadedVolume`'s new fields, `_viewer_headroom_bytes`, `_decimation_for`.

```bash
ruff check . && ruff format .
git add gui/viewers.py gui/widgets/ tests/ docs/Usage.md docs/Codebase.md
git commit -m "feat: decimate oversized volumes on read in the 3-D viewer

VTK uploads the whole array, so streaming cannot help this path; decimation
can. The stride is the smallest power of two bringing the volume within the
machine's headroom, spacing is scaled to match so the render stays physically
correct, and the factor is shown in the viewer. Also drops the redundant
full-size copy the percentile clim was making."
```

---

## Re-measure gate

**Before starting Wave 2**, re-run the STO2 figures. Wave 1 rewrote the peak model of five stages, so every number in the spec's table is now stale.

- [ ] Run each stage's estimator against the STO2 parameters and record the new `peak_bytes`, the same way commit `08513f5` recorded the originals.
- [ ] Compare each against the 8 GB floor's headroom (~3.6 GB) and the 16 GB laptop's (~7.2 GB).
- [ ] **Report the table to Albert and stop.** A stage now under the floor does not get its Wave 3 conversion built. This gate is the reason the waves are ordered this way; skipping it and building all four conversions anyway wastes exactly the work the ordering exists to avoid.

The gate's output is a decision about which of Tasks 9-12 survive. Do not proceed past it unattended.

---

## Wave 2 — Machinery

### Task 5: Halo and second axis for `iter_blocks`, plus the simple reductions

**Files:**
- Modify: `dfxm/common/volumeio.py:32-57` (`_layers_per_block`, `iter_blocks`), append the reductions after `neumaier_sum`
- Test: `tests/test_common_volumeio.py`
- Docs: `docs/Codebase.md`

**Interfaces:**
- Produces:
  - `iter_blocks(dset, *, budget_bytes, axis=0)` — unchanged behaviour; `axis` now accepts 0 or 1.
  - `iter_with_context(blocks, *, trailing=1)` — takes any ascending `(slice, array)` stream and yields `(interior, window, within)`: `window` is the block with the next block's first `trailing` rows appended, `interior` is the block's own range in source coordinates, and `within` indexes the interior inside `window`.
  - `dataset_blocks(dset, *, budget_bytes, axis=0) -> Iterator[np.ndarray]`
  - `stream_mean(blocks) -> float` — mean of finite values, or `nan` for none.
  - `stream_minmax(blocks) -> tuple[float, float]` — of finite values, `(nan, nan)` for none.
- Consumes: `neumaier_sum` (existing).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_common_volumeio.py`:

```python
def test_iter_with_context_appends_the_next_block_head():
    data = np.arange(20 * 3 * 4, dtype=np.float64).reshape(20, 3, 4)
    seen = np.zeros(20, dtype=int)
    windows = []
    blocks = volumeio.iter_blocks(data, budget_bytes=data.nbytes // 4)
    for interior, window, within in volumeio.iter_with_context(blocks, trailing=1):
        assert np.array_equal(window[within], data[interior])
        seen[interior] += 1
        windows.append((interior, window))
    assert np.array_equal(seen, np.ones(20, dtype=int)), "interiors must tile exactly once"
    # Every block but the last carries one row of the next.
    for interior, window in windows[:-1]:
        assert window.shape[0] == (interior.stop - interior.start) + 1
        assert np.array_equal(window[-1], data[interior.stop])
    last_interior, last_window = windows[-1]
    assert last_window.shape[0] == last_interior.stop - last_interior.start


def test_iter_with_context_works_on_a_generated_stream():
    """The point of taking a stream, not a dataset: generated blocks work too."""
    data = np.arange(12 * 2 * 2, dtype=np.float64).reshape(12, 2, 2)

    def generated():
        for start in range(0, 12, 5):
            stop = min(start + 5, 12)
            yield slice(start, stop), data[start:stop] * 2.0

    rebuilt = np.concatenate(
        [w[i] for _sl, w, i in volumeio.iter_with_context(generated(), trailing=1)], axis=0
    )
    assert np.array_equal(rebuilt, data * 2.0)


def test_iter_with_context_single_block():
    data = np.zeros((4, 2, 2))
    items = list(volumeio.iter_with_context(volumeio.iter_blocks(data, budget_bytes=1 << 30)))
    assert len(items) == 1
    interior, window, within = items[0]
    assert interior == slice(0, 4) and window.shape[0] == 4 and within == slice(0, 4)


def test_iter_blocks_axis_1():
    data = np.arange(4 * 12 * 3, dtype=np.float64).reshape(4, 12, 3)
    rebuilt = np.concatenate(
        [block for _sl, block in volumeio.iter_blocks(data, budget_bytes=data.nbytes // 3, axis=1)],
        axis=1,
    )
    assert np.array_equal(rebuilt, data)


def test_stream_mean_matches_nanmean_closely_and_is_budget_independent():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(30, 8, 8))
    data[data > 2] = np.nan
    means = [
        volumeio.stream_mean(volumeio.dataset_blocks(data, budget_bytes=data.nbytes // d))
        for d in (1, 3, 7, 1000)
    ]
    assert len({m.hex() for m in means}) == 1, "mean must not depend on the budget"
    assert means[0] == pytest.approx(float(np.nanmean(data)), rel=1e-12)


def test_stream_minmax_ignores_non_finite():
    data = np.array([[[1.0, np.nan]], [[np.inf, -3.0]], [[5.0, 2.0]]])
    lo, hi = volumeio.stream_minmax(volumeio.dataset_blocks(data, budget_bytes=8))
    assert (lo, hi) == (-3.0, 5.0)


def test_stream_mean_of_nothing_is_nan():
    data = np.full((4, 2, 2), np.nan)
    assert np.isnan(volumeio.stream_mean(volumeio.dataset_blocks(data, budget_bytes=16)))
```

The `hex()` comparison in the mean test is deliberate — `==` on floats would pass for values that differ in the last bit, which is precisely what this guarantee is about.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_common_volumeio.py -k "context or axis_1 or stream_" -v`
Expected: FAIL — `AttributeError: module 'dfxm.common.volumeio' has no attribute 'iter_with_context'`

- [ ] **Step 3: Implement the iterator changes**

Replace `_layers_per_block` and `iter_blocks` in `dfxm/common/volumeio.py`:

```python
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


def iter_with_context(blocks, *, trailing: int = 1):
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

    Memory cost is *trailing* rows above the block itself, so a budget is
    exceeded by that much; with the default of one row against any realistic
    block that is negligible, but it is stated because an unstated
    approximation in a memory-budget module is how budgets stop being trusted.
    """
    previous = None
    for sl, block in blocks:
        if previous is not None:
            prev_sl, prev_block = previous
            head = block[:trailing]
            window = np.concatenate([prev_block, head], axis=0) if head.size else prev_block
            yield prev_sl, window, slice(0, prev_block.shape[0])
        previous = (sl, block)
    if previous is not None:
        prev_sl, prev_block = previous
        yield prev_sl, prev_block, slice(0, prev_block.shape[0])
```

`_layers_per_block` needs no edit — it already takes `axis` and indexes `dset.shape[axis]`; confirm that and move on.

Note what `iter_with_context` does *not* do: it never re-reads. It holds one block plus the next block's head, which is why it works identically over an HDF5 dataset and over a generator that can only be walked forwards.

- [ ] **Step 4: Implement the reductions**

Append after `neumaier_sum`:

```python
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
```

`stream_minmax` may use `np.min`/`np.max` freely: min and max are partition-invariant in the strict sense the module requires — unlike summation, the answer genuinely does not depend on the grouping.

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_common_volumeio.py -q`
Expected: PASS, including every pre-existing test — `iter_blocks`'s existing two-tuple contract must be untouched.

- [ ] **Step 6: Update docs and commit**

`docs/Codebase.md`: the new signatures under `dfxm/common/volumeio.py`.

```bash
ruff check . && ruff format .
git add dfxm/common/volumeio.py tests/test_common_volumeio.py docs/Codebase.md
git commit -m "feat: block context, axis-1 iteration, and streaming mean/minmax

iter_with_context re-yields a block stream with each block carrying the next
block's first rows, which is what an operation with a forward-looking local
dependency needs to stream: linear interpolation and map_coordinates(order=1)
both read the row after the one they land on. It takes a stream rather than a
dataset because the blocks needing context are often generated and cannot be
re-indexed to widen a read.

axis=1 is what an in-plane median needs. The reductions take an iterable of
arrays for the same reason: the statistics they serve are over generated
blocks that never hit disk."
```

---

### Task 6: `stream_quantile` — an exact streaming order statistic

Colour limits come from `np.percentile` over the finite values. An approximation would shift every existing figure's colours, so this must return the same number — which means reproducing numpy's interpolation, not merely getting close.

**Files:**
- Modify: `dfxm/common/volumeio.py` (append after `stream_minmax`)
- Test: `tests/test_common_volumeio.py`
- Docs: `docs/Codebase.md`

**Interfaces:**
- Produces: `stream_quantile(make_blocks: Callable[[], Iterable[np.ndarray]], q: float) -> float`. Takes a **factory** — a zero-argument callable returning a fresh iterable — because the algorithm traverses three times.
- Consumes: `stream_minmax` (Task 5).

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("q", [0.0, 1.0, 25.0, 50.0, 99.0, 100.0])
def test_stream_quantile_matches_numpy_exactly(q):
    rng = np.random.default_rng(7)
    data = rng.normal(size=(40, 6, 6))
    data[data > 1.8] = np.nan
    finite = data[np.isfinite(data)]
    got = volumeio.stream_quantile(
        lambda: volumeio.dataset_blocks(data, budget_bytes=data.nbytes // 5), q
    )
    assert got == float(np.percentile(finite, q)), "must be bit-equal, not merely close"


def test_stream_quantile_is_budget_independent():
    rng = np.random.default_rng(11)
    data = rng.normal(size=(33, 5, 5))
    values = [
        volumeio.stream_quantile(
            lambda d=d: volumeio.dataset_blocks(data, budget_bytes=max(1, data.nbytes // d)), 99.0
        )
        for d in (1, 2, 7, 10_000)
    ]
    assert len({v.hex() for v in values}) == 1


def test_stream_quantile_handles_constant_data():
    data = np.full((10, 3, 3), 4.25)
    got = volumeio.stream_quantile(
        lambda: volumeio.dataset_blocks(data, budget_bytes=72), 50.0
    )
    assert got == 4.25


def test_stream_quantile_of_nothing_is_nan():
    data = np.full((4, 2, 2), np.nan)
    got = volumeio.stream_quantile(lambda: volumeio.dataset_blocks(data, budget_bytes=16), 50.0)
    assert np.isnan(got)


def test_stream_quantile_single_finite_value():
    data = np.full((4, 2, 2), np.nan)
    data[1, 1, 1] = 2.5
    got = volumeio.stream_quantile(lambda: volumeio.dataset_blocks(data, budget_bytes=16), 90.0)
    assert got == 2.5
```

The constant-data test is the one that catches a naive histogram implementation: with `lo == hi` the bin width is zero and every division blows up.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_common_volumeio.py -k stream_quantile -v`
Expected: FAIL — `AttributeError: module 'dfxm.common.volumeio' has no attribute 'stream_quantile'`

- [ ] **Step 3: Implement**

```python
_QUANTILE_BINS = 1 << 16
_QUANTILE_EXACT_CAP = 1 << 20


def _lerp(lo: float, hi: float, t: float) -> float:
    """numpy's linear interpolation for percentiles, reproduced bit-for-bit.

    numpy switches formula at t >= 0.5 for numerical stability; matching it is
    the difference between "the same number" and "a very close number", and
    the whole point of an exact quantile is the former.
    """
    result = lo + (hi - lo) * t
    if t >= 0.5 and hi != lo:
        result = hi - (hi - lo) * (1.0 - t)
    return float(result)


def stream_quantile(make_blocks, q: float) -> float:
    """The *q*-th percentile of the finite values, exactly, in bounded memory.

    Returns what ``np.percentile(finite_values, q)`` returns — not an estimate.
    Colour limits are computed this way, so an approximation would shift every
    existing figure's colours; exactness is the requirement.

    Three passes. Pass 1 takes the finite min, max and count. Pass 2
    histograms into the range and locates the bin holding the target rank,
    narrowing and repeating while that bin holds too many values to sort. Pass
    3 collects the survivors — a small array — and selects exactly.

    *make_blocks* is a zero-argument callable returning a fresh iterable of
    arrays, because the algorithm traverses more than once.
    """
    lo, hi = stream_minmax(make_blocks())
    if not np.isfinite(lo):
        return float("nan")
    count = 0
    for block in make_blocks():
        count += int(np.count_nonzero(np.isfinite(block)))
    if count == 1 or lo == hi:
        return float(lo)

    # numpy's rank convention for the "linear" method.
    pos = (q / 100.0) * (count - 1)
    rank_lo = int(np.floor(pos))
    frac = pos - rank_lo
    rank_hi = min(rank_lo + 1, count - 1)

    lo_val = _select_rank(make_blocks, rank_lo, lo, hi)
    hi_val = lo_val if rank_hi == rank_lo else _select_rank(make_blocks, rank_hi, lo, hi)
    if frac == 0.0:
        return float(lo_val)
    return _lerp(lo_val, hi_val, frac)


def _select_rank(make_blocks, rank: int, lo: float, hi: float) -> float:
    """The *rank*-th smallest finite value (0-based), by histogram refinement."""
    below = 0  # count of finite values strictly below `lo`
    while True:
        if lo == hi:
            return float(lo)
        edges = np.linspace(lo, hi, _QUANTILE_BINS + 1)
        counts = np.zeros(_QUANTILE_BINS, dtype=np.int64)
        for block in make_blocks():
            finite = block[np.isfinite(block)]
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
        if in_bin <= _QUANTILE_EXACT_CAP:
            survivors = []
            bin_lo, bin_hi = float(edges[bin_index]), float(edges[bin_index + 1])
            last = bin_index == _QUANTILE_BINS - 1
            for block in make_blocks():
                finite = block[np.isfinite(block)]
                mask = (finite >= bin_lo) & ((finite <= bin_hi) if last else (finite < bin_hi))
                if mask.any():
                    survivors.append(finite[mask])
            values = np.sort(np.concatenate(survivors))
            return float(values[target - before])
        # Too many to sort: narrow to this bin and go again.
        below += before
        lo, hi = float(edges[bin_index]), float(edges[bin_index + 1])
```

Three subtleties worth stating, because each is a plausible wrong answer rather than a crash. The `side="right"` in `searchsorted` on the edges plus the final-bin inclusive upper bound reproduces `np.histogram`'s convention, where the last bin includes its right edge and all others are half-open — get this wrong and the maximum value falls outside every bin. The `below` accumulator carries the count of values excluded by earlier narrowings, so `target` stays in the current window's coordinates. And refinement terminates because each round strictly narrows `[lo, hi]`; the `lo == hi` guard at the top handles the degenerate case where a bin's edges collapse under floating-point, which is what a large run of identical values produces.

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_common_volumeio.py -k stream_quantile -v`
Expected: PASS (9 tests including the parametrised ones)

If the exact-match test fails on the interpolated percentiles, the cause is almost certainly `_lerp` not matching this numpy version's `_lerp`. Read `numpy/lib/_function_base_impl.py::_lerp` in the installed numpy and mirror it — do not relax the test to `approx`, which would discard the property the task exists to provide.

- [ ] **Step 5: Update docs and commit**

```bash
ruff check . && ruff format .
git add dfxm/common/volumeio.py tests/test_common_volumeio.py docs/Codebase.md
git commit -m "feat: exact streaming quantile for colour limits

Colour limits come from np.percentile over finite values, so an approximate
streaming quantile would shift every existing figure's colours. This returns
the same number: min/max and count, then histogram refinement to isolate the
bin holding the target rank, then exact selection among the survivors."
```

---

### Task 7: `align_volume_streamed`

The fixed alignment chain, block by block, calling the existing step functions so there stays exactly one implementation of the arithmetic.

**Files:**
- Modify: `dfxm/common/alignment.py:62-96` (`apply_samy_shifts_to_volume`), `:99-131` (`interpolate_to_uniform_z`), `:176-212` (`AlignedVolume`, `align_volume`); append `StreamedAlignment` and `align_volume_streamed`
- Test: `tests/test_common_alignment.py`
- Docs: `docs/Codebase.md`

**Interfaces:**
- Produces:
  - `apply_samy_shifts_to_volume(volume, samy, scale_x, samy_direction=1, ref_samy=None, *, pad=None)` — `pad` is `(pad_left, pad_right)`; `None` computes it from `samy` exactly as today.
  - `interpolate_to_uniform_z(volume, samz, ref_samz=None, *, z_uniform=None)` — `None` derives the grid from `samz` exactly as today.
  - `StreamedAlignment` dataclass with `shape`, `dtype`, `z_uniform_um`, `scale_z_um`, `pad_left`, `center_offset`, and `blocks` — a **factory**, callable repeatedly, each call returning a fresh `Iterator[tuple[slice, np.ndarray]]` in ascending output-Z order.
  - `align_volume_streamed(dset, samy, samz, *, scale_x, samy_direction=1, roi_x=None, roi_y=None, take_abs=False, center_method=None, budget_bytes, scratch_dir=None) -> StreamedAlignment`
- Consumes: `volumeio.stream_mean`, `volumeio.stream_minmax`, `volumeio.stream_quantile`, `volumeio.scratch_array`.

- [ ] **Step 1: Write the failing parity test**

This is the task's whole point, so the test is written first and is the acceptance criterion:

```python
import numpy as np
import pytest

from dfxm.common import alignment as A


def _synthetic(nz=17, ny=6, nx=9, seed=3):
    rng = np.random.default_rng(seed)
    vol = rng.normal(size=(nz, ny, nx))
    vol[vol > 1.9] = np.nan
    samy = np.cumsum(rng.normal(scale=0.002, size=nz))
    samz = np.sort(rng.normal(scale=0.01, size=nz))
    return vol, samy, samz


@pytest.mark.parametrize("center_method", [None, "mean", "median"])
@pytest.mark.parametrize("divisor", [1, 3, 8, 10_000])
def test_streamed_alignment_matches_in_core(center_method, divisor):
    vol, samy, samz = _synthetic()
    kwargs = dict(
        scale_x=0.15,
        samy_direction=1,
        roi_x=(1, 8),
        roi_y=(0, 5),
        take_abs=False,
        center_method=center_method,
    )
    reference = A.align_volume(vol, samy, samz, **kwargs)
    streamed = A.align_volume_streamed(
        vol, samy, samz, budget_bytes=max(1, vol.nbytes // divisor), **kwargs
    )
    assert streamed.shape == reference.data.shape
    assert streamed.pad_left == reference.pad_left
    assert np.array_equal(streamed.z_uniform_um, reference.z_uniform_um)
    assert streamed.scale_z_um == reference.scale_z_um
    rebuilt = np.empty(streamed.shape, dtype=streamed.dtype)
    for zsl, block in streamed.blocks():
        rebuilt[zsl] = block
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


def test_streamed_blocks_factory_can_be_traversed_twice():
    vol, samy, samz = _synthetic()
    streamed = A.align_volume_streamed(
        vol, samy, samz, scale_x=0.15, budget_bytes=vol.nbytes // 4
    )
    first = np.concatenate([b for _sl, b in streamed.blocks()], axis=0)
    second = np.concatenate([b for _sl, b in streamed.blocks()], axis=0)
    assert np.array_equal(first, second, equal_nan=True)


def test_streamed_shape_known_before_reading():
    """A writer must be able to size its output before a voxel is read."""
    vol, samy, samz = _synthetic()
    streamed = A.align_volume_streamed(
        vol, samy, samz, scale_x=0.15, roi_x=(2, 7), budget_bytes=64
    )
    assert streamed.shape[1] == 6  # full Y
    assert streamed.shape[2] == 5 + streamed.pad_left + streamed.pad_right


def test_block_samy_slice_does_not_shrink_the_canvas():
    """The trap: a per-block samy slice implies a narrower pad than the global one."""
    vol, samy, samz = _synthetic(nz=21)
    samy = samy + np.linspace(0, 0.05, len(samy))  # a strong monotone drift
    reference = A.align_volume(vol, samy, samz, scale_x=0.15)
    streamed = A.align_volume_streamed(
        vol, samy, samz, scale_x=0.15, budget_bytes=vol.nbytes // 9
    )
    assert streamed.shape[2] == reference.data.shape[2]
```

The last test is the sharpest trap in the whole phase: `apply_samy_shifts_to_volume` derives the canvas width from `np.min`/`np.max` of the `samy` it is handed, so a per-block slice of a drifting `samy` produces a different width for every block. A strong monotone drift makes the failure loud.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_common_alignment.py -k streamed -v`
Expected: FAIL — `AttributeError: module 'dfxm.common.alignment' has no attribute 'align_volume_streamed'`

- [ ] **Step 3: Add the two compatible parameters**

In `apply_samy_shifts_to_volume`, replace the pad computation:

```python
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
    ...  # the remainder is unchanged
```

In `interpolate_to_uniform_z`, accept an explicit grid:

```python
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
```

with the body's grid construction guarded by `if z_uniform is None:` and the existing `n_use == 1` early return kept for the derived case only (an explicit grid means the caller already decided the output length). Keep every other line as-is.

- [ ] **Step 4: Implement the streaming alignment**

Append to `alignment.py`:

```python
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
    blocks: "Callable[[], Iterator[tuple[slice, np.ndarray]]]"


def _z_grid(samz: np.ndarray, n_layers: int, ref_samz: float | None):
    """The uniform Z grid and its step, from motor positions alone."""
    n_use = min(n_layers, len(samz))
    ref = float(samz[0]) if ref_samz is None else float(ref_samz)
    z_um = (np.asarray(samz[:n_use]) - ref) * 1000.0
    median_step = float(np.median(np.abs(np.diff(z_um)))) if n_use > 1 else 1.0
    if median_step < 1e-6:
        median_step = 1.0
    if n_use == 1:
        return z_um, z_um, median_step
    z_min, z_max = float(z_um.min()), float(z_um.max())
    n_uniform = max(2, int(np.round((z_max - z_min) / median_step)) + 1)
    z_uniform = np.linspace(z_min, z_max, n_uniform)
    scale_z = float(z_uniform[1] - z_uniform[0]) if n_uniform > 1 else median_step
    return z_um, z_uniform, scale_z


def _input_span(z_um: np.ndarray, z_target: np.ndarray) -> tuple[int, int]:
    """Input layer indices whose values linear interpolation of *z_target* reads.

    Returns a half-open ``(start, stop)``. Handles a decreasing or unsorted
    samz by spanning the full index range implied by the sorted order — larger
    reads, never a wrong answer.
    """
    order = np.argsort(z_um)
    ordered = z_um[order]
    lo = int(np.searchsorted(ordered, float(np.min(z_target)), side="right")) - 1
    hi = int(np.searchsorted(ordered, float(np.max(z_target)), side="left")) + 1
    lo = max(0, min(lo, len(ordered) - 1))
    hi = max(lo + 1, min(hi + 1, len(ordered)))
    used = order[lo:hi]
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
    in explicitly.

    Centring costs an extra traversal, because the statistic is over the
    *aligned* volume and cannot be precomputed from the source. ``median``
    costs three, so when the volume will not fit the budget its aligned blocks
    are cached to ``scratch_dir`` and the quantile's passes read from there.
    """
    from dfxm.common import volumeio

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
    dtype = np.dtype(np.float64)

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
            if len(samy) > 0:
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
        method = center_method.lower()
        if method == "mean":
            offset = volumeio.stream_mean(b for _sl, b in _blocks())
        elif method == "median":
            fits = nz * per_out_layer <= budget_bytes
            if fits or scratch_dir is None:
                offset = volumeio.stream_quantile(lambda: (b for _sl, b in _blocks()), 50.0)
            else:
                with volumeio.scratch_array(shape, dtype, dirpath=scratch_dir) as cache:
                    for zsl, block in _blocks():
                        cache[zsl] = block
                    offset = volumeio.stream_quantile(
                        lambda: volumeio.dataset_blocks(cache, budget_bytes=budget_bytes), 50.0
                    )
        else:
            raise ValueError(
                f"center_around_zero: unknown method {center_method!r} (expected mean/median)"
            )

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
```

Two notes for the implementer. `center_around_zero` computes the mean over `data[np.isfinite(data)]` via `np.nanmean`; `stream_mean` computes it via compensated summation, so the two differ by roughly 1 ulp — **this is the accepted drift from spec decision 3**, and the parity test above must therefore compare `align_volume` against a version that uses the same statistic. Reconcile by having `center_around_zero` call `volumeio.stream_mean([valid])` for the mean case and `volumeio.stream_quantile(lambda: [valid], 50.0)` for the median case, so both paths share one definition and the parity test is exact rather than approximate. Make that edit as part of this step; it is the single change that keeps "bit-identical across budgets" true rather than nearly true.

Second: `scratch_dir` comes from `RunPlan.scratch_dir`. When it is `None` and the volume does not fit, the code falls back to three alignment passes rather than failing — slower, never failed, per the project's governing rule.

- [ ] **Step 5: Reimplement `align_volume` as the façade**

```python
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
        budget_bytes=volume.nbytes * 8 + (1 << 20),
    )
    data = np.empty(streamed.shape, dtype=streamed.dtype)
    for zsl, block in streamed.blocks():
        data[zsl] = block
    return AlignedVolume(
        data, streamed.z_uniform_um, streamed.scale_z_um, streamed.pad_left, streamed.center_offset
    )
```

The generous `budget_bytes` forces a single block, so the in-core path stays one pass over one array — the fast-path guard, expressed in code rather than hoped for.

`AlignedVolume` gains nothing; keep it as-is so its consumers are untouched.

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest tests/test_common_alignment.py -q`
Expected: PASS, including every pre-existing alignment test. Those are the real regression suite here: `align_volume` is used by four stages and its behaviour must not move.

Then the broader sweep, since `align_volume` is shared: `python3 -m pytest tests/test_stage_visualize.py tests/test_stage_paraview.py tests/test_stage_slices.py tests/test_stage_rocking.py -q`

- [ ] **Step 7: Update docs and commit**

`docs/Codebase.md`: `StreamedAlignment`, `align_volume_streamed`, the new `pad` and `z_uniform` parameters, and the note that `align_volume` is now a façade. `docs/Usage.md`: no user-visible change yet — no stage calls the streamed path until Wave 3.

```bash
ruff check . && ruff format .
git add dfxm/common/alignment.py dfxm/common/volumeio.py tests/ docs/Codebase.md
git commit -m "feat: streaming Z-blocked alignment

align_volume_streamed runs the fixed chain block by block, calling the same
step functions in the same order. The global quantities those steps would
otherwise re-derive per block — the samy canvas pad and the uniform Z grid —
are computed once from the motor arrays and passed in explicitly, which is
what keeps every block on one canvas.

align_volume is now the in-core facade that drains the stream, so there is
one implementation rather than two to keep in step. center_around_zero routes
its mean and median through the same compensated reductions, which is the
~1 ulp drift accepted in the design."
```

---

### Task 8: The peak-RSS harness

An estimator that models a peak wrongly is this project's established failure mode. A `budget_bytes` parameter proves the streaming code path runs; only measurement proves the peak dropped.

**Files:**
- Create: `tests/peak_rss.py`
- Modify: `dfxm/runner.py` (add a public `pid` property to `StageRunner`)
- Test: `tests/test_runner_hints.py` (for the `pid` property)
- Docs: `docs/Codebase.md`

**Interfaces:**
- Produces: `tests.peak_rss.measure_peak_rss(target: str, params: dict, *, interval: float = 0.02, timeout: float = 300.0) -> tuple[object, int]` returning `(result, peak_rss_bytes)`. Raises `RuntimeError` on stage failure or timeout.
- Produces: `StageRunner.pid -> int | None`.
- Consumes: `dfxm.runner.StageRunner`, `psutil`.

- [ ] **Step 1: Write the failing test for `pid`**

Add to `tests/test_runner_hints.py`:

```python
def test_stage_runner_exposes_child_pid():
    from dfxm.runner import StageRunner

    runner = StageRunner("tests.peak_rss:_sleepy_target", {"seconds": 0.2})
    assert runner.pid is None, "no child before start()"
    runner.start()
    try:
        assert isinstance(runner.pid, int) and runner.pid > 0
    finally:
        runner.cancel()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_runner_hints.py -k child_pid -v`
Expected: FAIL — `AttributeError: 'StageRunner' object has no attribute 'pid'`

- [ ] **Step 3: Add the property**

In `dfxm/runner.py`, in the `# -- state ---` block beside `finished`:

```python
    @property
    def pid(self) -> int | None:
        """The child's PID once started, else None. Lets a caller watch its memory."""
        return self._proc.pid if self._proc is not None else None
```

- [ ] **Step 4: Write the harness**

Create `tests/peak_rss.py`:

```python
"""Measure a stage's peak resident memory (not a pytest file).

`budget_bytes` proves the streaming code path runs. It does not prove the
peak dropped: a stage can stream its read and then materialise a float64 copy
anyway, which is exactly the class of mistake the phase-1-4 estimators made.
This samples the real child process, so it measures the thing rather than the
model of it.

`dfxm/runner.py` already spawns every stage in a child, so there is nothing to
build — only to watch.
"""

from __future__ import annotations

import time

import psutil

from dfxm.runner import StageRunner


def _sleepy_target(params: dict, progress=None):
    """A do-nothing stage, for testing the harness itself."""
    time.sleep(float(params.get("seconds", 0.1)))
    return {"ok": True}


def _hungry_target(params: dict, progress=None):
    """Allocate roughly `mib` MiB and hold it, for testing the harness itself."""
    import numpy as np

    block = np.ones(int(float(params.get("mib", 64)) * (1 << 20) // 8), dtype=np.float64)
    time.sleep(0.2)
    return {"sum": float(block[0])}


def measure_peak_rss(
    target: str, params: dict, *, interval: float = 0.02, timeout: float = 300.0
) -> tuple[object, int]:
    """Run *target* in a child and return ``(result, peak_rss_bytes)``.

    The sampling interval bounds the error: an allocation that lives for less
    than one interval can be missed entirely, so a peak measured here is a
    lower bound on the true peak. That direction is the safe one — a stage
    that passes may be better than measured, never worse.
    """
    runner = StageRunner(target, params)
    runner.start()
    proc = psutil.Process(runner.pid)
    peak = 0
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                peak = max(peak, int(proc.memory_info().rss))
            except psutil.NoSuchProcess:
                pass
            runner.poll()
            if runner.finished:
                break
            if not runner.is_alive():
                runner.poll()
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"{target} did not finish within {timeout}s")
            time.sleep(interval)
    finally:
        if runner.is_alive():
            runner.cancel()
        runner.join(timeout=5.0)
    runner.poll()
    if runner.failure is not None:
        raise RuntimeError(f"{target} failed: {runner.failure}")
    return runner.result, peak


def assert_peak_under(target: str, params: dict, limit_bytes: int, **kwargs) -> object:
    """Run *target* and require its peak RSS stayed under *limit_bytes*."""
    result, peak = measure_peak_rss(target, params, **kwargs)
    assert peak <= limit_bytes, (
        f"{target} peaked at {peak / (1 << 20):.1f} MiB, over the "
        f"{limit_bytes / (1 << 20):.1f} MiB limit"
    )
    return result
```

Check `StageRunner`'s public accessors before writing `runner.failure` and `runner.result` — read `dfxm/runner.py:182` onward and use whatever the `# -- state --` block actually exposes.

- [ ] **Step 5: Test the harness against itself**

A measurement harness that always returns zero would pass every later task silently. Add to `tests/test_runner_hints.py`:

```python
def test_peak_rss_sees_a_large_allocation():
    from tests.peak_rss import measure_peak_rss

    _small, baseline = measure_peak_rss("tests.peak_rss:_hungry_target", {"mib": 8})
    _big, hungry = measure_peak_rss("tests.peak_rss:_hungry_target", {"mib": 256})
    assert hungry - baseline > 128 * (1 << 20), (
        f"harness did not observe the allocation: {baseline} -> {hungry}"
    )
```

Comparing two runs rather than asserting an absolute figure keeps the test independent of the interpreter's own footprint, which varies with the numpy build.

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest tests/test_runner_hints.py -q`
Expected: PASS

If `test_peak_rss_sees_a_large_allocation` is flaky, lower `interval` to `0.005` before weakening the threshold — a sampling harness that misses a 256 MiB allocation is not fit for the purpose it exists for.

- [ ] **Step 7: Update docs and commit**

```bash
ruff check . && ruff format .
git add tests/peak_rss.py tests/test_runner_hints.py dfxm/runner.py docs/Codebase.md
git commit -m "test: peak-RSS measurement harness for stage runs

A budget parameter proves the streaming path runs; it does not prove the peak
dropped. This samples the real child process that runner.py already spawns,
and is self-tested against a known allocation so it cannot pass by measuring
nothing."
```

---

## Wave 3 — Conversions

**Which of Tasks 9-12 to build is decided at the re-measure gate.** Each assumes Waves 1 and 2 are merged.

### Task 9: `paraview` — stream the piece writer

`save_volumes_as_pvti` builds a full boolean `valid_mask` plus a `np.where`-cleaned float64 copy of every field before writing a single piece. Since pieces are Z-ranges, both can be done per piece.

**Files:**
- Modify: `dfxm/stages/paraview.py:484-...` (`save_volumes_as_pvti`), `:686-741` (`_process_mosaicity`), `:744-...` (`_process_strain`)
- Test: `tests/test_stage_paraview.py`
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Produces: `save_volumes_streamed(providers: dict[str, StreamedAlignment], spacing, output_path_pvti, *, origin=(0.0, 0.0, 0.0), n_pieces=16, compression=False, replace_nan=True, write_valid_mask=True, nan_sentinel=None) -> dict` — same return contract as `save_volumes_as_pvti` (`dimensions_xyz`, `spacing_um_xyz`, `origin_um_xyz`, `n_pieces`, `fields`).
- Consumes: `alignment.StreamedAlignment` (Task 7), `volumeio.stream_minmax` (Task 5).

- [ ] **Step 1: Write the failing equivalence test**

```python
def test_streamed_pvti_matches_in_core(tmp_path):
    """Streaming the piece writer changes nothing about the pieces."""
    import numpy as np

    from dfxm.stages import paraview

    fields = {"chi_Center_of_mass": _synthetic_volume(seed=1),
              "mu_FWHM": _synthetic_volume(seed=2)}
    reference_dir = tmp_path / "ref"
    streamed_dir = tmp_path / "str"
    reference_dir.mkdir()
    streamed_dir.mkdir()

    ref_info = paraview.save_volumes_as_pvti(
        fields, (0.1, 0.2, 0.3), str(reference_dir / "v.pvti"), n_pieces=4
    )
    providers = {name: _as_streamed(vol, budget_bytes=vol.nbytes // 5)
                 for name, vol in fields.items()}
    str_info = paraview.save_volumes_streamed(
        providers, (0.1, 0.2, 0.3), str(streamed_dir / "v.pvti"), n_pieces=4
    )

    assert str_info == ref_info
    for ref_piece in sorted(reference_dir.rglob("*.vti")):
        rel = ref_piece.relative_to(reference_dir)
        assert (streamed_dir / rel).read_bytes() == ref_piece.read_bytes(), rel
```

Comparing the written `.vti` bytes is the strongest available statement: the sentinel, the valid mask, the dtype downcast and the piece extents all have to agree, and any one of them drifting shows up here.

`_as_streamed` is a helper this test needs — write it in the test module, wrapping an in-memory array in `align_volume_streamed` with an identity alignment (empty `samy`/`samz`) so the provider protocol is exercised without the alignment maths confusing the comparison.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_stage_paraview.py -k streamed_pvti -v`
Expected: FAIL — `AttributeError: module 'dfxm.stages.paraview' has no attribute 'save_volumes_streamed'`

- [ ] **Step 3: Implement `save_volumes_streamed`**

Two passes, no scratch. Pass 1 needs only two scalars — the sentinel's global min/max across all fields, and the invalid-voxel count for `nan_fraction_overall`. Pass 2 writes pieces.

First the two helpers. Fields must be walked together, because the valid mask combines all of them:

```python
def _fields_in_lockstep(providers):
    """Walk every provider's blocks together, yielding ``(z_slice, {name: block})``.

    All providers are built from one budget over one shape, so they block
    identically — but that is checked rather than assumed, because a silent
    misalignment here would mix one field's voxels with another's.
    """
    iterators = {name: pv.blocks() for name, pv in providers.items()}
    while True:
        batch = {}
        for name, it in iterators.items():
            batch[name] = next(it, None)
        if all(item is None for item in batch.values()):
            return
        if any(item is None for item in batch.values()):
            raise ValueError("providers disagree on block count")
        ranges = {(sl.start, sl.stop) for sl, _block in batch.values()}
        if len(ranges) != 1:
            raise ValueError(f"providers disagree on block ranges: {ranges}")
        start, stop = ranges.pop()
        yield slice(start, stop), {name: block for name, (_sl, block) in batch.items()}


def _count_invalid(providers) -> int:
    """Voxels where ANY field is non-finite — the in-core ``valid_mask`` semantics."""
    invalid = 0
    for _sl, fields in _fields_in_lockstep(providers):
        mask = None
        for block in fields.values():
            finite = np.isfinite(block)
            mask = finite if mask is None else (mask & finite)
        invalid += int(np.count_nonzero(~mask))
    return invalid


class _SlabReader:
    """Serve ascending, possibly overlapping Z ranges from one forward pass.

    ``compute_piece_extents_z`` makes adjacent pieces share one Z index, so
    requests overlap and a consume-and-discard reader would drop a row it
    still needs. This keeps the buffered blocks that any pending request can
    still reach, and no more.
    """

    def __init__(self, providers) -> None:
        self._stream = _fields_in_lockstep(providers)
        self._buffer: list = []
        self._exhausted = False

    def slab(self, z0: int, z1: int) -> dict:
        """Fields over the half-open range ``[z0, z1)``, concatenated."""
        self._buffer = [(sl, f) for sl, f in self._buffer if sl.stop > z0]
        while not self._exhausted and (not self._buffer or self._buffer[-1][0].stop < z1):
            item = next(self._stream, None)
            if item is None:
                self._exhausted = True
                break
            self._buffer.append(item)
        if not self._buffer:
            raise ValueError(f"no blocks cover Z range [{z0}, {z1})")
        out = {}
        for name in self._buffer[0][1]:
            parts = []
            for sl, fields in self._buffer:
                lo, hi = max(sl.start, z0), min(sl.stop, z1)
                if hi > lo:
                    parts.append(fields[name][lo - sl.start : hi - sl.start])
            out[name] = np.concatenate(parts, axis=0)
        return out
```

Then the writer itself:

```python
def save_volumes_streamed(
    providers: dict,
    spacing: tuple,
    output_path_pvti: str,
    *,
    origin: tuple = (0.0, 0.0, 0.0),
    n_pieces: int = 16,
    compression: bool = False,
    replace_nan: bool = True,
    write_valid_mask: bool = True,
    nan_sentinel: float | None = None,
) -> dict:
    """Write a partitioned VTI dataset from streamed fields, one piece at a time.

    ``providers`` maps field name to a ``StreamedAlignment``. Every provider
    must share one shape — they are co-registered by construction, so a
    mismatch is a bug rather than a case to handle.

    Two passes. The first computes only what a piece cannot know locally: the
    NaN sentinel's global range and the overall invalid fraction. The second
    builds each piece's valid mask and cleaned arrays from that piece's Z-slab
    alone, so peak memory is one piece per field rather than one volume per
    field.
    """
    from dfxm.common import volumeio

    if not providers:
        raise ValueError("No volumes to save")
    if not output_path_pvti.endswith(".pvti"):
        raise ValueError(f"output_path_pvti must end with .pvti: {output_path_pvti}")

    shapes = {name: tuple(pv.shape) for name, pv in providers.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"All volumes must share the same shape, got {shapes}")
    nz, ny, nx = next(iter(shapes.values()))

    # --- pass 1: the two things a piece cannot compute for itself ---
    sentinel = None
    nan_fraction_overall = 0.0
    if replace_nan or write_valid_mask:
        total = nz * ny * nx
        nan_fraction_overall = (_count_invalid(providers) / total) if total else 0.0
    if replace_nan:
        if nan_sentinel is not None:
            sentinel = float(nan_sentinel)
        else:
            global_min, global_max = np.inf, -np.inf
            for pv in providers.values():
                lo, hi = volumeio.stream_minmax(block for _sl, block in pv.blocks())
                if np.isfinite(lo):
                    global_min = min(global_min, lo)
                    global_max = max(global_max, hi)
            sentinel = (
                -1e30
                if not np.isfinite(global_min)
                else global_min - 1000.0 * max(global_max - global_min, 1.0)
            )

    # --- pass 2: one piece at a time ---
    reader = _SlabReader(providers)
    extents = compute_piece_extents_z(nz, n_pieces)
    for piece_index, extent in enumerate(extents):
        z0, z1 = extent
        fields = reader.slab(z0, z1 + 1)  # extents are inclusive
        mask = None
        for block in fields.values():
            finite = np.isfinite(block)
            mask = finite if mask is None else (mask & finite)
        cleaned = {
            name: (np.where(np.isfinite(block), block, sentinel) if replace_nan else block)
            for name, block in fields.items()
        }
        if write_valid_mask:
            cleaned["valid_mask"] = mask.astype(SAVE_DTYPE)
        write_piece_vti(cleaned, extent, ...)  # match the existing call's remaining arguments
```

The `write_piece_vti(...)` call is the one place to copy rather than invent: read the existing call inside `save_volumes_as_pvti` and pass exactly the same remaining arguments (output directory, piece index, spacing, origin, compression). It already takes an explicit Z sub-extent, so it needs no change — but note whether it expects the field arrays sized to the piece or to the whole volume, and match that. Then reproduce the `.pvti` manifest writing that follows the piece loop in `save_volumes_as_pvti`, and return the same dict.

Three details that decide whether the byte-comparison test passes. The sentinel uses **per-field** `np.isfinite` for cleaning but the **combined** mask for `valid_mask` — that asymmetry is in the in-core code and must be preserved. `nan_fraction_overall` counts voxels where *any* field is non-finite, matching `valid_mask &= np.isfinite(v)`. And piece extents are **inclusive**, so the slab request is `[z0, z1 + 1)`.

- [ ] **Step 4: Wire `_process_mosaicity` and `_process_strain` to it**

Replace the `processed[name] = data` accumulation with a `providers[name] = A.align_volume_streamed(...)` construction, and call `save_volumes_streamed`. The `abs_mosa_fwhm` flag maps to `take_abs`, `center_mosa_com` + `center_method` map to `center_method`, and `roi_x`/`roi_y` pass straight through. Read the current body once more before editing — the `is_com` / `is_fwhm` branching must land on the right provider arguments.

`raw_detector_origin` needs `z_positions`, which is `provider.z_uniform_um` from any field — take it from the first provider rather than the old `z_positions` local.

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_stage_paraview.py -q`
Expected: PASS, including the byte-comparison test.

- [ ] **Step 6: Add the peak-RSS assertion**

```python
def test_paraview_peak_stays_under_budget(tmp_path):
    from tests.peak_rss import assert_peak_under

    params = _paraview_params(tmp_path, layers=64, ny=128, nx=128)
    # One float64 volume is 8 MiB; a bounded run must not need many of them.
    assert_peak_under("dfxm.stages.paraview:run", params, limit_bytes=400 * (1 << 20))
```

Size the synthetic input so the unconverted code would clearly exceed the limit and the converted code clearly does not — then verify that claim by running the test against the previous commit and watching it fail. A peak-RSS test that passes both before and after proves nothing.

- [ ] **Step 7: Update docs and commit**

`docs/Usage.md`: the paraview section gains a line that a run too large for the machine streams per piece — slower, identical output. `docs/Codebase.md`: `save_volumes_streamed` and the helpers.

```bash
ruff check . && ruff format .
git add dfxm/stages/paraview.py tests/test_stage_paraview.py docs/
git commit -m "perf: write PVTI pieces from streamed fields

The piece writer built a full boolean valid_mask and a np.where-cleaned
float64 copy of every field before writing a single piece. Pieces are Z
ranges, so both are per-piece work. One pass computes the two things a piece
cannot know locally - the sentinel's global range and the overall invalid
fraction - and the second writes pieces from Z-slabs.

Pieces are byte-identical to the in-core writer's."
```

---

### Task 10: `visualize` — streamed alignment

**Files:**
- Modify: `dfxm/stages/visualize.py:532-543` (`_align`), `:544-...` (`_process_dataset`), `:656-728` (`run`)
- Test: `tests/test_stage_visualize.py`
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `alignment.align_volume_streamed` (Task 7), `volumeio.stream_quantile` / `stream_minmax` (Tasks 5-6), `tests.peak_rss` (Task 8).

- [ ] **Step 1: Establish what `_process_dataset` actually needs**

Read `visualize.py:544` onward before writing any code and write down, in the task report, which of these `_process_dataset` requires: the whole aligned array, per-layer access in order, or a global statistic. `visualize` renders per-layer figures, so per-layer access in ascending Z is the expected answer — but confirm it rather than assume it, because that single fact decides whether this task is a rewrite or a rewiring.

- [ ] **Step 2: Write the failing equivalence test**

```python
def test_visualize_streamed_matches_in_core(tmp_path):
    """Every rendered layer is identical whether the volume streamed or not."""
    from dfxm.stages import visualize

    params = _visualize_params(tmp_path)
    reference = visualize.run({**params, "output_dir": str(tmp_path / "ref")})
    streamed = visualize.run(
        {**params, "output_dir": str(tmp_path / "str"), "_budget_bytes": 1 << 16}
    )
    assert [d.name for d in streamed.datasets] == [d.name for d in reference.datasets]
    for ref_png in sorted((tmp_path / "ref").rglob("*.png")):
        rel = ref_png.relative_to(tmp_path / "ref")
        assert (tmp_path / "str" / rel).read_bytes() == ref_png.read_bytes(), rel
```

The `_budget_bytes` key is a test seam. Decide deliberately whether it is a real (undocumented, underscore-prefixed) parameter the GUI can inject or a monkeypatch point, and be consistent with how the phase-1-4 work injects `plot_style` at `gui/stage_view.py:405`. Match that pattern; do not invent a second convention.

- [ ] **Step 3: Convert `run`'s two sections**

Both sections follow one shape: build a `StreamedAlignment` for the field, derive `vmin`/`vmax` from streaming reductions, then feed `_process_dataset` a block factory instead of an array.

`_align` gains a streaming sibling and keeps its in-core form, because `aligned_field` and the `figures` closure both call it and both genuinely want an array — `aligned_field` hands its result to the 3-D viewer:

```python
def _align_streamed(dset, samy, samz, *, scale_x, samy_direction, roi_x, roi_y, budget_bytes):
    """The streaming counterpart of ``_align``; same arguments, same order."""
    streamed = A.align_volume_streamed(
        dset,
        samy,
        samz,
        scale_x=scale_x,
        samy_direction=samy_direction,
        roi_x=roi_x,
        roi_y=roi_y,
        budget_bytes=budget_bytes,
    )
    return streamed, streamed.z_uniform_um, streamed.scale_z_um
```

The mosaicity loop then becomes — note the field is opened once and its blocks re-walked, so the file handle is bound to the iteration:

```python
        names = mosa_field_names(mosa_file)
        samy, samz = _read_motors(raw_root, p["mosa_pattern"], p["samy_path"], p["samz_path"])
        for i, name in enumerate(names):
            progress(0.1 + 0.4 * i / max(1, len(names)), f"mosaicity: {name}")
            title, cbar, group = _display_info(name)
            cmap = resolve_cmap(style, group)
            with h5py.File(mosa_file, "r") as f:
                dset = _mosa_dataset(f, name)
                if dset is None:
                    continue
                streamed, z_pos, scale_z = _align_streamed(
                    dset, samy, samz,
                    scale_x=scale_x, samy_direction=samy_dir,
                    roi_x=roi_x, roi_y=roi_y, budget_bytes=budget_bytes,
                )
                if "Center_of_mass" in name:
                    blocks, vmin, vmax = _center_com_and_range_streamed(
                        streamed, p["center_method"], float(p["range_pct"])
                    )
                else:
                    blocks = streamed.blocks
                    vmin, vmax = _colorbar_range_streamed(streamed)
                vmin, vmax, clim_note = apply_round_clim(vmin, vmax, style)
                if clim_note:
                    progress(0.1 + 0.4 * i / max(1, len(names)), f"{name}: {clim_note}")
                prod = _process_dataset(
                    blocks, streamed.shape, z_pos, scale_z, name, vmin, vmax, cmap,
                    title, cbar, p, out_dir, style=style, group=group,
                )
            if clim_note:
                prod.notes.append(clim_note)
            result.datasets.append(prod)
```

This needs `_mosa_dataset(f, name)` — the open-dataset half of `load_mosa_field` from Task 2. Factor it there and have `load_mosa_field` call it, so the name-matching convention lives in one place.

The three clim helpers get streaming siblings; each keeps its in-core form for the replot paths:

```python
def _colorbar_range_streamed(streamed) -> tuple[float, float]:
    lo = V.stream_quantile(lambda: (b for _sl, b in streamed.blocks()), 1.0)
    hi = V.stream_quantile(lambda: (b for _sl, b in streamed.blocks()), 99.0)
    return lo, hi


def _symmetric_range_streamed(streamed) -> tuple[float, float]:
    lo, hi = V.stream_minmax(b for _sl, b in streamed.blocks())
    limit = max(abs(lo), abs(hi))
    return -limit, limit


def _center_com_and_range_streamed(streamed, center_method, range_pct):
    """Centre a CoM field and return ``(blocks_factory, vmin, vmax)``."""
    method = center_method.lower()
    if method == "mean":
        offset = V.stream_mean(b for _sl, b in streamed.blocks())
    else:
        offset = V.stream_quantile(lambda: (b for _sl, b in streamed.blocks()), 50.0)

    def blocks():
        return ((sl, b - offset) for sl, b in streamed.blocks())

    lo, hi = V.stream_minmax(b for _sl, b in blocks())
    limit = max(abs(lo), abs(hi)) * (range_pct / 100.0)
    return blocks, -limit, limit
```

Read the three in-core originals before writing these — the `range_pct` convention in particular is asserted by existing tests, and the sibling has to reproduce it exactly rather than plausibly.

`_process_dataset` is the one signature that changes: it takes `(blocks, shape, ...)` where it took `(data, ...)`. Step 1 established whether it needs per-layer access in order; if it does, the change is mechanical — iterate `blocks()` and render each layer as it arrives, tracking the absolute Z index from the yielded slice. The strain section takes the same treatment with `_symmetric_range_streamed`.

- [ ] **Step 4: Run the tests, then add the peak assertion**

Run: `python3 -m pytest tests/test_stage_visualize.py tests/test_gui_viewers.py -q`

Then add a `assert_peak_under("dfxm.stages.visualize:run", ...)` test on the same pattern as Task 9 Step 6, including the check that it fails against the previous commit.

- [ ] **Step 5: Update docs and commit**

```bash
ruff check . && ruff format .
git add dfxm/stages/visualize.py tests/ docs/
git commit -m "perf: stream the aligned volume in visualize

Renders per-layer figures from Z-blocks rather than from a whole aligned
volume, with colour limits from the streaming reductions. Rendered PNGs are
byte-identical to the in-core path."
```

---

### Task 11: `slices` — Z-blocked gather

`map_coordinates(..., order=1)` reads the eight voxels bracketing each sample, two of them in Z. A Z-block plus the next block's first row can compute exactly the samples landing inside it, and the output is a small 2-D image per plane — so one pass over Z serves every plane at once.

**Files:**
- Modify: `dfxm/stages/slices.py:731-...` (`prepare_volume`), `:600-630` (`sample_plane`), `:1379-...` (the volumes loop), `:1150-1200` (`estimate`, to set `chunkable=True`)
- Test: `tests/test_stage_slices.py`
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `alignment.align_volume_streamed` (Task 7), `volumeio.iter_with_context` and `volumeio.iter_blocks` (Task 5).

- [ ] **Step 1: Write the failing equivalence test**

```python
def test_slices_streamed_gather_matches_in_core(tmp_path):
    import numpy as np

    from dfxm.stages import slices

    volume = _synthetic_volume(nz=24, ny=20, nx=20, seed=5)
    plane = _plane_spec()  # normal, up, origin, half_u, half_v, du, dv
    reference = slices.sample_plane(_prep_from(volume), **plane)[0]
    streamed = slices.sample_plane_streamed(
        _as_streamed(volume, budget_bytes=volume.nbytes // 6), **plane
    )[0]
    assert np.array_equal(streamed, reference, equal_nan=True)


def test_slices_gather_is_budget_independent(tmp_path):
    from tests.equivalence import assert_budget_independent

    volume = _synthetic_volume(nz=24, ny=20, nx=20, seed=6)
    plane = _plane_spec()
    assert_budget_independent(
        lambda dset, budget_bytes: slices.sample_plane_streamed(
            _as_streamed(dset, budget_bytes=budget_bytes), **plane
        )[0],
        volume,
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_stage_slices.py -k streamed_gather -v`
Expected: FAIL — `AttributeError: module 'dfxm.stages.slices' has no attribute 'sample_plane_streamed'`

- [ ] **Step 3: Implement the blocked gather**

First factor the coordinate arithmetic out of `sample_plane` so both samplers share it verbatim — the coordinates are not what changes:

```python
def _plane_coords(prep, plane_origin, u_hat, v_hat, half_u, half_v, du, dv):
    """Voxel coordinates ``(3, nv, nu)`` for one plane, plus its axes."""
    nu = max(1, int(np.round(2.0 * half_u / du)) + 1)
    nv = max(1, int(np.round(2.0 * half_v / dv)) + 1)
    u_um = np.linspace(-half_u, half_u, nu)
    v_um = np.linspace(-half_v, half_v, nv)
    uu, vv = np.meshgrid(u_um, v_um)
    pts = (
        np.asarray(plane_origin, np.float64)[None, None, :]
        + uu[..., None] * u_hat[None, None, :]
        + vv[..., None] * v_hat[None, None, :]
    )
    i = pts[..., 0] / prep["scale_x"] + prep["x_ref_shift_px"]
    j = pts[..., 1] / prep["scale_y"] + prep["y_ref_shift_px"]
    k = (pts[..., 2] - prep.get("z_ref_shift_um", 0.0)) / prep["scale_z"]
    return np.stack([k, j, i], axis=0), u_um, v_um


def sample_plane(prep, plane_origin, u_hat, v_hat, half_u, half_v, du, dv):
    """Sample one plane centred at plane_origin (X,Y,Z µm). Returns (slice, u_um, v_um)."""
    coords, u_um, v_um = _plane_coords(prep, plane_origin, u_hat, v_hat, half_u, half_v, du, dv)
    s = map_coordinates(prep["data"], coords, order=1, mode="constant", cval=np.nan)
    return s.astype(np.float32), u_um, v_um
```

Then the streamed sampler:

```python
def sample_plane_streamed(prep, plane_origin, u_hat, v_hat, half_u, half_v, du, dv):
    """The same plane, gathered from Z-blocks instead of a whole volume.

    ``map_coordinates`` with ``order=1`` reads layers ``floor(k)`` and
    ``floor(k) + 1``, so a sample belongs to the block whose interior contains
    ``floor(k)``, and that block needs one row of its successor to be
    self-sufficient — which is what ``iter_with_context`` supplies. Samples
    whose ``floor(k)`` lands in no block are outside the volume in Z and keep
    the NaN the output starts as, exactly as ``cval=np.nan`` produces in-core.
    """
    from dfxm.common.volumeio import iter_with_context

    coords, u_um, v_um = _plane_coords(prep, plane_origin, u_hat, v_hat, half_u, half_v, du, dv)
    out = np.full(coords.shape[1:], np.nan, dtype=np.float64)
    k, j, i = coords[0], coords[1], coords[2]
    home = np.floor(k)  # NaN and inf coordinates match no block, and stay NaN

    for interior, window, _within in iter_with_context(prep["blocks"](), trailing=1):
        sel = (home >= interior.start) & (home < interior.stop)
        if not sel.any():
            continue
        local = np.stack([k[sel] - interior.start, j[sel], i[sel]], axis=0)
        out[sel] = map_coordinates(window, local, order=1, mode="constant", cval=np.nan)
    return out.astype(np.float32), u_um, v_um
```

Why this is bit-identical rather than merely close, since the reasoning is the whole justification for the task. A sample assigned to block `[z0, z1)` has `floor(k)` in `[z0, z1-1]`, so it reads rows `floor(k)` and `floor(k)+1`, both of which lie within the window `[z0, z1]` that `trailing=1` provides — the same two rows, with the same weights, as the in-core call. Non-final blocks therefore never touch a boundary condition at all. The final block has no successor row, so its window ends exactly where the volume ends, which means scipy applies the identical out-of-bounds rule there that it applies in-core. And a sample with `floor(k) < 0` or `>= nz` matches no block and keeps its NaN, which is what in-core `cval` gives it.

The one thing not to "simplify": use `np.floor(k)` for the assignment, never `np.round` or an integer cast. `int(-0.5)` truncates toward zero and would assign a sample above the volume's floor to block 0.

- [ ] **Step 4: Wire `prepare_volume` to return a provider**

`prepare_volume` returns a dict whose `"data"` key holds the array. Replace that with a `"blocks"` factory, which `sample_plane_streamed` calls. Both source kinds can supply one, and neither may leave an HDF5 file closed under a generator that is still being walked — so the file's lifetime is bound to the volume's, with an `ExitStack` owned by the caller:

```python
def prepare_volume(cfg, p, scale_x, scale_y, samy_dir, style=None, *, stack, budget_bytes):
    """Load and (if stacked) align one volume, resolving render style per kind.

    Returns a dict whose ``blocks`` key is a factory yielding ``(z_slice,
    array)`` over the prepared volume. *stack* is an ``ExitStack`` owned by
    the caller: the HDF5 file must stay open for as long as the blocks are
    walked, and must close when the caller is finished with this volume.
    """
    from dfxm.common.volumeio import iter_blocks

    kind, source = cfg["kind"], cfg["source"]
    extra = {}
    f = stack.enter_context(h5py.File(cfg["h5_path"], "r"))
    dset = f[cfg["dataset_path"]]
    if source == "stacked":
        samy, samz, _ = _motors(cfg, p)
        streamed = A.align_volume_streamed(
            dset,
            samy,
            samz,
            scale_x=scale_x,
            samy_direction=samy_dir,
            roi_x=cfg.get("roi_x"),
            roi_y=cfg.get("roi_y"),
            take_abs=(kind == "mosa_fwhm" and bool(p["abs_fwhm"])),
            center_method=None,  # slices centres itself below, including midrange
            budget_bytes=budget_bytes,
        )
        blocks = streamed.blocks
        scale_z = streamed.scale_z_um
        sx, sy = scale_x, scale_y
        x_ref = y_ref = 0.0
        z_ref = float(cfg.get("z_ref_shift_um", 0.0))
    else:  # aligned — already co-registered, no alignment step
        extra = dict(f.attrs)
        sx = float(extra.get("scale_x_um_per_px", scale_x))
        sy = float(extra.get("scale_y_um_per_px", scale_y))
        scale_z = float(extra.get("scale_z_um_per_px", 1.0))
        x_ref = float(cfg.get("x_ref_shift_px", 0))
        y_ref = float(cfg.get("y_ref_shift_px", 0))
        z_ref = float(cfg.get("z_ref_shift_um", 0.0))

        def blocks():
            for sl, block in iter_blocks(dset, budget_bytes=budget_bytes):
                yield sl, block.astype(np.float64)
```

The centring and clim block then works over `blocks()` instead of `data`, and each of the four helpers gains a streaming sibling:

```python
    center_method = p["center_method"].lower()
    if kind in ("mosa_com", "strain"):
        if center_method == "midrange":
            lo, hi = V.stream_minmax(b for _sl, b in blocks())
            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * float(p["range_pct"]) / 100.0
            auto_vmin, auto_vmax = -half, half
        else:
            center = (
                V.stream_mean(b for _sl, b in blocks())
                if center_method == "mean"
                else V.stream_quantile(lambda: (b for _sl, b in blocks()), 50.0)
            )
            limit = max(
                abs(V.stream_quantile(lambda: (b - center for _sl, b in blocks()), 0.0)),
                abs(V.stream_quantile(lambda: (b - center for _sl, b in blocks()), 100.0)),
            )
            auto_vmin, auto_vmax = -limit, limit
    else:  # mosa_fwhm / raw_*
        center = 0.0
        auto_vmin = V.stream_quantile(lambda: (b for _sl, b in blocks()), 1.0)
        auto_vmax = V.stream_quantile(lambda: (b for _sl, b in blocks()), 99.0)
```

with the centring itself folded into the factory the sampler receives, so the subtraction happens once per block rather than over a materialised volume:

```python
    prepared = blocks if not center else (lambda: ((sl, b - center) for sl, b in blocks()))
```

Before writing this, **read `_center_offset`, `_symmetric_range`, `_percentile_range` and `_midrange_clim` and match their exact semantics** — the code above reproduces what those four appear to do from their call site, but each is a real function with its own edge-case handling (empty input, all-NaN, the `range_pct` convention), and a streaming sibling that diverges on an edge case is a silent wrong answer rather than a failure. Where a helper's in-core form is still used by a replot path that already holds an array, keep it; add the sibling beside it rather than replacing it.

`midrange` stays in `slices.py`. It is the only stage offering it, and promoting it into `alignment.py` would put a slices-specific convention into shared code.

Finally the caller, which now owns the file's lifetime — and in doing so subsumes Task 3's explicit `prep = None`:

```python
        for vi, cfg in enumerate(volumes):
            progress(0.1 + 0.85 * vi / len(volumes), f"slicing {cfg['kind']} {cfg['dataset_path']}")
            with contextlib.ExitStack() as stack:
                try:
                    prep = prepare_volume(
                        cfg, p, scale_x, scale_y, samy_dir,
                        style=style, stack=stack, budget_bytes=budget_bytes,
                    )
                except (KeyError, OSError, ValueError) as exc:
                    result.skipped.append(f"{cfg['dataset_path']}: {exc}")
                    continue
                ...  # the plane loop, unchanged except sample_plane -> sample_plane_streamed
```

Keep `sample_plane` and its in-core path: `figures()` and the replot helpers call it with an array they already hold, and forcing those through a block factory would be churn for no memory saved.

- [ ] **Step 5: Correct the estimator's `chunkable` flag**

In `slices.estimate`, change `chunkable=False` to `True` and rewrite the docstring line "Not chunkable: alignment is a whole-volume operation" — it is the claim this task disproves. The peak model itself is Task 13's job; do not touch it here beyond the flag and that sentence.

- [ ] **Step 6: Run the tests and add the peak assertion**

Run: `python3 -m pytest tests/test_stage_slices.py tests/test_slices_marks.py tests/test_stage_estimates.py -q`

Add `assert_peak_under("dfxm.stages.slices:run", ...)` on the Task 9 pattern.

- [ ] **Step 7: Update docs and commit**

```bash
ruff check . && ruff format .
git add dfxm/stages/slices.py tests/ docs/
git commit -m "perf: gather oblique slices from Z-blocks

map_coordinates with order=1 reads the eight voxels bracketing each sample,
two of them in Z, so a Z-block plus the next block's first row computes the
samples landing inside it. The output is a small 2-D image per plane, so one
pass over Z serves every requested plane.

Corrects the estimator's chunkable=False: alignment is not a whole-volume
operation when blocked along the axis it actually runs over."
```

---

### Task 12: `rocking`'s percentile and `matched`'s in-plane median

Two small, unrelated conversions, kept in one task because neither carries its own review weight.

**Files:**
- Modify: `dfxm/stages/rocking.py:606-610` (`_colorbar_range`), `:1028-1042` (`_replot_default_clim`)
- Modify: `dfxm/stages/matched.py:263-278` (`load_pco_ff_frame`), `:318-...` (`estimate`, to set `chunkable=True`)
- Test: `tests/test_stage_rocking.py`, `tests/test_stage_matched.py`
- Docs: `docs/Codebase.md`

**Interfaces:**
- Consumes: `volumeio.stream_quantile` (Task 6), `volumeio.iter_blocks(axis=1)` (Task 5).

- [ ] **Step 1: Write the failing tests**

```python
def test_rocking_replot_clim_matches_in_core(tmp_path):
    import h5py
    import numpy as np

    from dfxm.stages import rocking

    path = str(tmp_path / "aligned.h5")
    rng = np.random.default_rng(4)
    volume = rng.normal(size=(20, 16, 16))
    volume[volume > 1.7] = np.nan
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=volume)
    finite = volume[np.isfinite(volume)]
    expected = (float(np.percentile(finite, 1)), float(np.percentile(finite, 99)))
    with h5py.File(path, "r") as f:
        got = rocking._replot_default_clim(f["sum_intensity"], {}, _default_style())
    assert got[0] == pytest.approx(expected[0]) and got[1] == pytest.approx(expected[1])


def test_matched_blocked_median_matches_whole_stack(tmp_path):
    import h5py
    import numpy as np

    from dfxm.stages import matched

    rng = np.random.default_rng(9)
    stack = rng.normal(size=(11, 12, 14))
    path = str(tmp_path / "scan.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("1.1/measurement/pco_ff", data=stack)
    background = np.nanmedian(stack, axis=0)
    expected = stack[3] - background
    expected[expected < 0] = np.nan
    got = matched.load_pco_ff_frame(path, "1.1/measurement/pco_ff", 3, budget_bytes=1024)
    assert np.array_equal(got, expected, equal_nan=True)
```

The rocking test uses `approx` because `apply_round_clim` may round; if `_replot_default_clim` returns the raw percentiles, tighten it to exact equality — exactness is the property Task 6 provides and it should be asserted wherever it survives.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_stage_rocking.py tests/test_stage_matched.py -k "replot_clim or blocked_median" -v`
Expected: FAIL — `load_pco_ff_frame() got an unexpected keyword argument 'budget_bytes'`

- [ ] **Step 3: Convert `rocking`**

```python
def _replot_default_clim(dataset, params: dict, style, *, budget_bytes: int | None = None) -> tuple[float, float]:
    """Compute the default clim for a cold replot the same way the run does.

    Streams the global percentiles rather than loading the volume: the
    quantile is exact, so a replot's colours are identical to the original
    PNGs' on any machine. Falls back to ``(0.0, 1.0)`` for an all-NaN volume.
    """
    from dfxm.common.volumeio import dataset_blocks, stream_quantile

    defaults = STAGE.defaults()
    pct_lo = float(params.get("cbar_pct_lo", defaults["cbar_pct_lo"]))
    pct_hi = float(params.get("cbar_pct_hi", defaults["cbar_pct_hi"]))
    budget = budget_bytes or (64 << 20)

    def blocks():
        return dataset_blocks(dataset, budget_bytes=budget)

    vmin = stream_quantile(blocks, pct_lo)
    vmax = stream_quantile(blocks, pct_hi)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (0.0, 1.0)
    vmin, vmax, _ = apply_round_clim(vmin, vmax, style)
    return vmin, vmax
```

`_colorbar_range` keeps its in-core form for the callers that already hold an array; only the replot path, which held a whole volume purely to take two percentiles, changes.

- [ ] **Step 4: Convert `matched`**

```python
def load_pco_ff_frame(h5_path, pco_ff_path, frame_index, *, budget_bytes: int | None = None):
    """Single frame, per-pixel median background subtracted, negatives -> NaN.

    The median is over the frame axis, so it is exact on any in-plane
    sub-block: ``stack[:, y0:y1, :]`` gives the same answer for those rows as
    the whole stack does. Blocking in-plane is therefore lossless, which is
    why this stage's estimator now reports ``chunkable=True``.
    """
    from dfxm.common.volumeio import iter_blocks

    with h5py.File(h5_path, "r") as f:
        if pco_ff_path not in f:
            return None
        ds = f[pco_ff_path]
        if ds.ndim == 2:
            return ds[:].astype(np.float64)
        if ds.ndim != 3:
            return None
        idx = min(int(frame_index), ds.shape[0] - 1)
        budget = budget_bytes or (256 << 20)
        out = np.empty((ds.shape[1], ds.shape[2]), dtype=np.float64)
        for ysl, block in iter_blocks(ds, budget_bytes=budget, axis=1):
            sub = block.astype(np.float64)
            background = np.nanmedian(sub, axis=0)
            out[ysl] = sub[idx] - background
    out[out < 0] = np.nan
    return out
```

The `out[out < 0] = np.nan` moves outside the file context and outside the loop, matching the in-core version's single application to the finished frame — applying it per block would be equivalent here, but keeping it whole keeps the diff honest about what changed.

Then set `chunkable=True` in `matched.estimate` and rewrite the docstring's "Not chunkable: an exact median needs the whole stack" to state that it needs the whole stack along the frame axis only.

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_stage_rocking.py tests/test_stage_matched.py tests/test_stage_estimates.py -q`
Expected: PASS

- [ ] **Step 6: Update docs and commit**

```bash
ruff check . && ruff format .
git add dfxm/stages/rocking.py dfxm/stages/matched.py tests/ docs/Codebase.md
git commit -m "perf: stream rocking's replot percentiles and matched's median

rocking loaded a whole volume to take two percentiles for a colour range;
the streaming quantile is exact, so replot colours are unchanged. matched's
median is over the frame axis, so it is exact on any in-plane sub-block -
correcting a chunkable=False the code contradicts."
```

---

## Wave 4 — Verification

### Task 13: Re-verify every estimator and run the STO2 canary

Every peak model in the seven estimators describes code that Waves 1-3 rewrote. Phases 1-4 shipped estimators whose models were wrong and needed three fix waves to correct; this task is where the same mistake gets caught instead of shipped.

**Files:**
- Modify: `dfxm/stages/strain.py`, `mosaicity.py`, `visualize.py`, `paraview.py`, `slices.py`, `rocking.py`, `matched.py` (the `estimate` functions and their docstrings)
- Test: `tests/test_stage_estimates.py`
- Docs: `docs/Usage.md`, `docs/Codebase.md`

- [ ] **Step 1: Re-derive each peak model from the rewritten `run()`**

For each of the seven stages, read the current `run()` body and write the peak model from what the code now does — not from what the old docstring says. Record, per stage, the high-water expression and the line that produces it. A model derived from the previous docstring is the exact failure this task exists to prevent.

One model gains a term it never had: a chunked run with `center_method="median"` caches its aligned blocks to scratch disk (Task 7), so for the stages offering that setting the estimate must report the **scratch requirement** as well as the memory peak. Without it, `advice.plan_run` cannot set `RunPlan.blocked`, and a machine short of disk discovers the problem halfway through a long run instead of before it starts. `CostEstimate` has no field for this today — add one (`scratch_bytes: int = 0`, defaulting to zero so every other estimator is unaffected) and have `plan_run` consult it in the check it already performs against `profile.disk_free`. This is the one place phase 5 widens a phase-1-4 interface; keep the widening to that.

- [ ] **Step 2: Check each model against measurement**

For each stage, use `tests/peak_rss.py` to measure the real peak on a synthetic input of known size, and compare against `estimate(params).peak_bytes` for the same input. Require agreement within a factor of 1.5 in either direction. A model that under-predicts is dangerous — it tells `plan_run` a run fits when it does not — so treat under-prediction as a failure even when the measured peak is comfortable.

Add these as tests in `tests/test_stage_estimates.py`, one per stage, so the agreement is enforced rather than checked once.

- [ ] **Step 3: Update the docstrings**

Each `estimate` docstring states the peak model in prose. Rewrite all seven. Where a `chunkable` flag changed (`slices`, `matched`), make sure the docstring's reasoning matches the new value — Tasks 11 and 12 changed the flag and one sentence; this step makes the whole docstring coherent.

- [ ] **Step 4: Run the memory-capped STO2 canary**

This is the phase's acceptance criterion and it is hand-run, not automated:

1. For each converted stage, run STO2 unconstrained and keep the output.
2. Re-run with `budget_bytes` forced to a value well under the stage's unconstrained peak — simulating the 8 GB floor — while `tests/peak_rss.py` watches.
3. Assert three things: the run completes; the measured peak stayed under the cap; and the output is bit-identical to the unconstrained run. For HDF5 products compare datasets with `np.array_equal(..., equal_nan=True)` rather than file bytes, since chunking and attribute order can differ harmlessly. For PNG and `.vti` products compare bytes.
4. Record the table — stage, unconstrained peak, capped peak, cap, identical yes/no — in the commit body, the way commit `08513f5` recorded the original figures.

Any stage failing point 3 is a phase-5 defect, not a tolerance to relax.

- [ ] **Step 5: Update the docs**

`docs/Usage.md`: a short section stating that stages adapt to the machine — a run too large for available memory streams rather than failing, takes longer, and produces the identical product; that the 3-D viewer is the one place coarsening happens, and it says so; and that products created before this change reproduce to within ~1 ulp rather than bit-for-bit, because the centring statistic is now compensated. That last point belongs in the user guide, not only in the spec: it is the one thing a user could otherwise discover by being confused.

`docs/Codebase.md`: confirm every function added across the phase is present and that removed ones (`save_stacked_volume`, `load_mosa_datasets`) are gone.

- [ ] **Step 6: Full verification and commit**

```bash
ruff check . && ruff format .
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
DISPLAY= python3 -u tests/gui_smoke.py
```

Both must be green — the suite with no failures, the smoke run through `[40]` with `[41]` retried once if it GL-flakes. Do not claim completion from a partial run.

```bash
git add dfxm/stages/ tests/test_stage_estimates.py docs/
git commit -m "fix: re-verify all seven estimators against the rewritten stages

Every peak model described code that phase 5 rewrote. Each is re-derived from
the current run() body and checked against a measured peak, with
under-prediction treated as a failure since it tells plan_run a run fits when
it does not.

Includes the memory-capped STO2 canary: <table>."
```

---

## Notes for the executor

**Order is not optional.** Wave 2's machinery is what Wave 3 converts onto, and the re-measure gate between Waves 1 and 2 decides which Wave 3 tasks exist at all. Do not start a Wave 3 task before the gate has been reported to Albert and answered.

**Every equivalence test must be non-vacuous.** `tests/equivalence.py` warns that a single-element `budgets` list is a vacuous pass. The same trap has a second form here: a synthetic volume small enough that every budget yields one block tests nothing. Size fixtures so at least the smallest budget forces several blocks, and assert the block count where it is cheap to do so.

**Peak-RSS tests must be shown to fail.** A `assert_peak_under` test that passes against both the converted and the unconverted code proves only that the limit was generous. For each one, run it against the previous commit and confirm it fails before committing it.

**Test fixture helpers are not specified, deliberately.** Names like `_synthetic_volume`, `_visualize_params` or `_plane_spec` appear in the test code above as calls, not definitions. Every one of those test modules already has its own synthetic-input helpers; find and reuse them rather than adding a parallel set. Where a module genuinely lacks one, build it from the nearest existing helper in that same file and keep its conventions — a second fixture style in a module is a cost paid by every future reader of it.

**The `~1 ulp` drift is intended.** If a test comparing against a pre-phase-5 stored product fails in the last bits of a centred quantity, that is spec decision 3 working, not a regression — update the stored expectation and say so in the commit. Any *other* difference is a defect.
