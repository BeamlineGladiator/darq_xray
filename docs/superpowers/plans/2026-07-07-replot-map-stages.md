# Replot for strain / mosaicity / rocking (+ ROI crop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the slices-only **Replot…** capability (cold re-render from disk with clim override + layer subset) to the **strain**, **mosaicity**, and **rocking** stages, and add a **pixel-index ROI crop** override to replot on all four stages (including the existing slices dialog).

**Architecture:** Add small Qt-free core primitives to `dfxm/common/figures.py` (`crop_roi_2d`, `render_volume_layer`, `ReplotGroup`) reused by every stage; give strain/mosaicity/rocking the same `replot_catalog` + `render_replot` public pair slices already exposes; add one generic 2-level `ReplotDialog` in the GUI plus ROI boxes on the existing `SliceReplotDialog`; un-gate the `Replot…` button for the three new stages. No stage h5 output format changes.

**Tech Stack:** Python 3.10, numpy, h5py, matplotlib (explicit `Figure`/Agg API — never pyplot), PySide6 (GUI only), pytest.

## Global Constraints

- **`dfxm/` stays Qt-free** — never import PySide6/pyvista in the core; all figure work lives in `dfxm/`.
- **Plotting via the explicit `Figure` API** — never `pyplot` or `matplotlib.use(...)`.
- **No stage h5 output format changes** — replot reads existing files only.
- **ROI convention = `(r0, r1, c0, c1)`** pixel bounds (rows then cols), matching strain's existing `roi` param; blank/None = full extent; crop is clamped per-array; an empty crop yields `None` (skipped, never raised).
- **clim override = `(vmin, vmax)`** where either may be `None` (keep stored); `clim=None` keeps stored limits.
- **Replot is re-plot, never re-analysis** — read stored layer → optional crop → optional clim → draw. It cannot recover data outside what the run stored.
- **Docs contract (same change):** any task that changes a stage's public functions or a viewer updates `docs/Usage.md` (user-facing) and `docs/Codebase.md` (code reference) in the SAME commit.
- **`ruff format` runs automatically** on Write/Edit (line length 100, double quotes). Run `ruff check .` before each commit.
- **Tests:** synthetic HDF5 fixtures under `tmp_path`; assert clim via `fig.axes[0].images[0].norm.vmin/vmax` and crop via the image array/extent.
- Full suite: `python3 -m pytest -q`. GUI smoke (not pytest): `python3 tests/gui_smoke.py`.

---

## File Structure

- `dfxm/common/figures.py` — **modify**: add `ReplotGroup`, `crop_roi_2d`, `render_volume_layer`; refactor `volume_layer_specs` to render through `render_volume_layer` (DRY).
- `dfxm/stages/mosaicity.py` — **modify**: add `replot_catalog` + `render_replot`.
- `dfxm/stages/rocking.py` — **modify**: add `replot_catalog` + `render_replot`.
- `dfxm/stages/strain.py` — **modify**: add `replot_catalog` + `render_replot` + `_rebuild_strain_map`.
- `dfxm/stages/slices.py` — **modify**: add `roi` to `render_replot` + `_rebuild_plane_figure`.
- `gui/widgets/replot_dialog.py` — **create**: generic 2-level `ReplotDialog`.
- `gui/widgets/slice_replot.py` — **modify**: add ROI boxes + `roi` pass-through.
- `gui/stage_view.py` — **modify**: un-gate `Replot…` button; per-stage `_on_replot` dispatch.
- Tests: `tests/test_figures_replot.py` (create), `tests/test_stage_mosaicity.py`, `tests/test_stage_rocking.py`, `tests/test_stage_strain.py`, `tests/test_stage_slices.py`, `tests/test_gui_replot_dialog.py` (create), `tests/gui_smoke.py`, `tests/test_stage_view_buttons.py` (create or extend existing stage_view test).
- Docs: `docs/Usage.md`, `docs/Codebase.md`.

---

### Task 1: Core replot primitives in `dfxm/common/figures.py`

**Files:**
- Modify: `dfxm/common/figures.py`
- Test: `tests/test_figures_replot.py` (create)

**Interfaces:**
- Consumes: existing `_load_layer(h5_path, dataset, z)`, `render.layer_figure(...)`, `dfxm.common.plotting.resolve_cmap`.
- Produces:
  - `@dataclass ReplotGroup(key: str, label: str, item_labels: list[str])`
  - `crop_roi_2d(layer: np.ndarray, roi: tuple[int,int,int,int] | None) -> np.ndarray | None`
  - `render_volume_layer(h5_path, dataset, z, *, cmap, cmap_group, title, cbar_label, sx, sy, vmin, vmax, style, clim=None, roi=None, z_um=None) -> Figure | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_figures_replot.py`:

```python
import h5py
import numpy as np
import pytest

from dfxm.common import figures as F


def _write_vol(path, key="/chi/Center of mass", shape=(2, 4, 5)):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=rng.standard_normal(shape).astype(np.float32))
    return key


def test_crop_roi_2d_slices_and_clamps():
    arr = np.arange(20, dtype=float).reshape(4, 5)
    # exact crop
    out = F.crop_roi_2d(arr, (1, 3, 0, 2))
    assert out.shape == (2, 2)
    assert np.array_equal(out, arr[1:3, 0:2])
    # None → unchanged (same object is fine)
    assert F.crop_roi_2d(arr, None) is arr
    # out-of-range bounds are clamped to the array
    assert F.crop_roi_2d(arr, (-5, 999, -5, 999)).shape == (4, 5)
    # empty crop → None
    assert F.crop_roi_2d(arr, (2, 2, 0, 5)) is None


def test_render_volume_layer_applies_clim_and_roi(tmp_path):
    h5 = tmp_path / "vol.h5"
    key = _write_vol(str(h5), shape=(2, 4, 5))
    fig = F.render_volume_layer(
        str(h5), key, 0,
        cmap="magma", cmap_group="mosa_com", title="t", cbar_label="c",
        sx=0.1, sy=0.2, vmin=0.0, vmax=1.0, style=None,
        clim=(-3.0, 3.0), roi=(1, 3, 0, 2),
    )
    im = fig.axes[0].images[0]
    assert im.norm.vmin == -3.0 and im.norm.vmax == 3.0
    # extent reflects the CROP (2 cols × 0.1, 2 rows × 0.2), origin at 0
    left, right, bottom, top = im.get_extent()
    assert (right, top) == pytest.approx((0.2, 0.4))
    assert (left, bottom) == (0.0, 0.0)


def test_render_volume_layer_empty_crop_returns_none(tmp_path):
    h5 = tmp_path / "vol.h5"
    key = _write_vol(str(h5), shape=(1, 4, 5))
    fig = F.render_volume_layer(
        str(h5), key, 0,
        cmap="magma", cmap_group=None, title="t", cbar_label="c",
        sx=0.1, sy=0.1, vmin=0.0, vmax=1.0, style=None,
        roi=(2, 2, 0, 5),
    )
    assert fig is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_figures_replot.py -v`
Expected: FAIL with `AttributeError: module 'dfxm.common.figures' has no attribute 'crop_roi_2d'`.

- [ ] **Step 3: Implement the primitives**

In `dfxm/common/figures.py`, add near the top (after existing imports; add `from dataclasses import dataclass, field` and `import numpy as np` if not present):

```python
@dataclass
class ReplotGroup:
    """One selectable group in a replot catalog: a dataset/product with N layers."""

    key: str  # in-file dataset key (mosaicity/rocking) or logical group id
    label: str  # tree display label
    item_labels: list[str] = field(default_factory=list)  # per-layer labels


def crop_roi_2d(layer, roi):
    """Crop a 2-D array to ``(r0, r1, c0, c1)`` pixel bounds, clamped to shape.

    ``roi=None`` returns *layer* unchanged. Returns ``None`` when the (clamped)
    crop is empty. Replot ROI is a reframe of stored data, never a recompute.
    """
    if roi is None:
        return layer
    r0, r1, c0, c1 = roi
    h, w = layer.shape[:2]
    r0 = max(0, min(int(r0), h))
    r1 = max(0, min(int(r1), h))
    c0 = max(0, min(int(c0), w))
    c1 = max(0, min(int(c1), w))
    if r1 <= r0 or c1 <= c0:
        return None
    return layer[r0:r1, c0:c1]


def _apply_clim(vmin, vmax, clim):
    if clim is None:
        return vmin, vmax
    lo, hi = clim
    return (lo if lo is not None else vmin, hi if hi is not None else vmax)


def render_volume_layer(
    h5_path,
    dataset,
    z,
    *,
    cmap,
    cmap_group,
    title,
    cbar_label,
    sx,
    sy,
    vmin,
    vmax,
    style,
    clim=None,
    roi=None,
    z_um=None,
):
    """Read one (Z,Y,X) layer, optionally crop + clim-override, return a map Figure.

    Returns ``None`` when the ROI crop is empty. Shared by ``volume_layer_specs``
    (export) and the mosaicity/rocking ``render_replot`` (cold replot).
    """
    from .plotting import resolve_cmap
    from . import render

    layer = _load_layer(h5_path, dataset, z)
    layer = crop_roi_2d(layer, roi)
    if layer is None:
        return None
    ext_x = layer.shape[1] * sx
    ext_y = layer.shape[0] * sy
    v0, v1 = _apply_clim(vmin, vmax, clim)
    zlabel = f"\nZ = {z_um[z]:.2f} µm" if z_um is not None else ""
    fig, _, _ = render.layer_figure(
        layer,
        v0,
        v1,
        resolve_cmap(style, cmap_group, fallback=cmap),
        ext_x,
        ext_y,
        f"{title}{zlabel} (layer {z})",
        cbar_label,
        style=style,
        group=cmap_group,
    )
    return fig
```

Then refactor `volume_layer_specs`'s inner `build(style)` (currently lines ~124-142) to delegate, so there is one render path:

```python
    def make(z):
        def build(style):
            fig = render_volume_layer(
                h5_path,
                dataset,
                z,
                cmap=cmap,
                cmap_group=cmap_group,
                title=title,
                cbar_label=cbar_label,
                sx=sx,
                sy=sy,
                vmin=vmin,
                vmax=vmax,
                style=style,
                z_um=z_um,
            )
            return fig

        return build
```

- [ ] **Step 4: Run tests + existing export tests to verify pass**

Run: `python3 -m pytest tests/test_figures_replot.py tests/test_figures_catalog.py tests/test_export_fidelity.py -q`
Expected: PASS (new tests pass; the `volume_layer_specs` refactor keeps export figures identical).

- [ ] **Step 5: Commit**

```bash
ruff check dfxm/common/figures.py tests/test_figures_replot.py
git add dfxm/common/figures.py tests/test_figures_replot.py
git commit -m "feat(core): replot primitives (crop_roi_2d, render_volume_layer, ReplotGroup)"
```

---

### Task 2: mosaicity `replot_catalog` + `render_replot`

**Files:**
- Modify: `dfxm/stages/mosaicity.py`
- Test: `tests/test_stage_mosaicity.py`
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `ReplotGroup`, `render_volume_layer` (Task 1); existing `_KEY_DISPLAY`, `_KEY_STEM`, `_streamed_clim`.
- Produces:
  - `mosaicity.replot_catalog(h5_path) -> list[ReplotGroup]` — one group per 3-D dataset key present.
  - `mosaicity.render_replot(h5_path, selections, style, clim, out_dir, roi=None, params=None) -> list[str]` — `selections` is `list[(key, item_idxs | None)]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stage_mosaicity.py`:

```python
def _write_stacked(path):
    import h5py
    import numpy as np

    rng = np.random.default_rng(3)
    with h5py.File(path, "w") as f:
        for key in ("/chi/Center of mass", "/chi/FWHM"):
            f.create_dataset(key, data=rng.standard_normal((2, 4, 5)).astype(np.float32))
        f.attrs["num_layers"] = 2
    return path


def test_mosaicity_replot_catalog_lists_datasets(tmp_path):
    h5 = str(tmp_path / "stacked.h5")
    _write_stacked(h5)
    cat = M.replot_catalog(h5)
    by_key = {g.key: g for g in cat}
    assert set(by_key) == {"/chi/Center of mass", "/chi/FWHM"}
    assert len(by_key["/chi/FWHM"].item_labels) == 2


def test_mosaicity_render_replot_writes_pngs_with_crop(tmp_path):
    import os

    h5 = str(tmp_path / "stacked.h5")
    _write_stacked(h5)
    out = str(tmp_path / "replots")
    written = M.render_replot(
        h5,
        [("/chi/Center of mass", [0]), ("/chi/FWHM", None)],
        style=None,
        clim=None,
        out_dir=out,
        roi=(0, 2, 0, 3),
    )
    assert len(written) == 1 + 2
    assert all(os.path.exists(p) for p in written)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_mosaicity.py -k replot -v`
Expected: FAIL with `AttributeError: module 'dfxm.stages.mosaicity' has no attribute 'replot_catalog'`.

- [ ] **Step 3: Implement**

In `dfxm/stages/mosaicity.py`, add (import the core helpers at top: `from dfxm.common.figures import ReplotGroup, render_volume_layer`):

```python
def replot_catalog(h5_path: str) -> list[ReplotGroup]:
    """List every 3-D mosaicity dataset in a stacked h5 as a replot group."""
    groups: list[ReplotGroup] = []
    with h5py.File(h5_path, "r") as f:
        for key in _KEY_DISPLAY:
            obj = f.get(key)
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                continue
            _grp, _cbar, title = _KEY_DISPLAY[key]
            n_z = obj.shape[0]
            groups.append(
                ReplotGroup(
                    key=key,
                    label=title,
                    item_labels=[f"layer {z}" for z in range(n_z)],
                )
            )
    return groups


def render_replot(h5_path, selections, style, clim, out_dir, roi=None, params=None) -> list[str]:
    """Re-render selected mosaicity map layers cold from a stacked h5.

    ``selections`` is ``list[(dataset_key, item_idxs | None)]`` (``None`` = all
    layers). ``clim`` overrides vmin/vmax; ``roi`` crops each layer (pixel bounds).
    PNGs are written under ``{out_dir}/{stem}/``; returns written paths.
    """
    params = params or {}
    px = float(params.get("pixel_size_x_um", 0.152))
    py = float(params.get("pixel_size_y_um", 0.385))
    written: list[str] = []
    with h5py.File(h5_path, "r") as f:
        for key, idxs in selections:
            obj = f.get(key)
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                continue
            group, cbar_label, title = _KEY_DISPLAY.get(key, (None, "(°)", key))
            stem = _KEY_STEM.get(key, key.lstrip("/").replace("/", "_").replace(" ", "_"))
            n_z = obj.shape[0]
            vmin, vmax = _streamed_clim(obj)
            layer_list = list(range(n_z)) if idxs is None else list(idxs)
            sub_dir = os.path.join(out_dir, stem)
            os.makedirs(sub_dir, exist_ok=True)
            for z in layer_list:
                if z < 0 or z >= n_z:
                    continue
                fig = render_volume_layer(
                    h5_path, key, z,
                    cmap="magma", cmap_group=group, title=title, cbar_label=cbar_label,
                    sx=px, sy=py, vmin=vmin, vmax=vmax, style=style, clim=clim, roi=roi,
                )
                if fig is None:
                    continue
                png = os.path.join(sub_dir, f"{stem}_layer_{z:04d}.png")
                fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
                written.append(png)
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_mosaicity.py -q`
Expected: PASS.

- [ ] **Step 5: Update docs + commit**

Add a mosaicity Replot note to `docs/Usage.md` (Replot section) and the two new functions to `docs/Codebase.md` (`dfxm/stages/mosaicity`). Then:

```bash
ruff check dfxm/stages/mosaicity.py tests/test_stage_mosaicity.py
git add dfxm/stages/mosaicity.py tests/test_stage_mosaicity.py docs/Usage.md docs/Codebase.md
git commit -m "feat(mosaicity): cold replot_catalog + render_replot with clim/ROI"
```

---

### Task 3: rocking `replot_catalog` + `render_replot`

**Files:**
- Modify: `dfxm/stages/rocking.py`
- Test: `tests/test_stage_rocking.py`
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `ReplotGroup`, `render_volume_layer` (Task 1); existing `_PRODUCT_DATASET`, `_PRODUCT_CBAR`, `_streamed_clim` (mosaicity's is module-local — rocking needs its own tiny streamed-clim or reuse `np.nanmin/nanmax` on the layer stack).
- Produces:
  - `rocking.replot_catalog(h5_path) -> list[ReplotGroup]`
  - `rocking.render_replot(h5_path, selections, style, clim, out_dir, roi=None, params=None) -> list[str]` — `selections` is `list[(dataset_key, item_idxs | None)]` where dataset_key ∈ {`sum_intensity`, `specific_frame`}.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stage_rocking.py`:

```python
def _write_aligned(path):
    import h5py
    import numpy as np

    rng = np.random.default_rng(5)
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=rng.standard_normal((2, 4, 5)).astype(np.float32))
        f.create_dataset("specific_frame", data=rng.standard_normal((2, 4, 5)).astype(np.float32))
        f.create_dataset("z_uniform_um", data=np.arange(2, dtype=np.float32))
        f.attrs["scale_x_um_per_px"] = 0.152
        f.attrs["scale_y_um_per_px"] = 0.385
    return path


def test_rocking_replot_catalog_lists_products(tmp_path):
    h5 = str(tmp_path / "aligned.h5")
    _write_aligned(h5)
    cat = R.replot_catalog(h5)
    keys = {g.key for g in cat}
    assert keys == {"sum_intensity", "specific_frame"}


def test_rocking_render_replot_writes_pngs_with_clim(tmp_path):
    import os

    h5 = str(tmp_path / "aligned.h5")
    _write_aligned(h5)
    out = str(tmp_path / "replots")
    written = R.render_replot(
        h5,
        [("sum_intensity", None)],
        style=None,
        clim=(0.0, 2.0),
        out_dir=out,
    )
    assert len(written) == 2
    assert all(os.path.exists(p) for p in written)
```

(Confirm the test module imports `rocking as R`; if it uses a different alias, match it.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_rocking.py -k replot -v`
Expected: FAIL with `AttributeError: ... 'rocking' has no attribute 'replot_catalog'`.

- [ ] **Step 3: Implement**

In `dfxm/stages/rocking.py` add (import `from dfxm.common.figures import ReplotGroup, render_volume_layer`). Map each in-file dataset key to a display title/cbar; reverse `_PRODUCT_DATASET` for titles:

```python
# in-file dataset key → (default title, default cbar) for cold replot
_DATASET_DISPLAY: dict[str, tuple[str, str]] = {
    "sum_intensity": ("Sum intensity", "Sum intensity (a.u.)"),
    "specific_frame": ("Specific Frame", "Intensity (a.u.)"),
}


def _layer_clim(dataset) -> tuple[float, float]:
    lo = float("inf")
    hi = float("-inf")
    for z in range(dataset.shape[0]):
        layer = dataset[z]
        finite = layer[np.isfinite(layer)]
        if finite.size:
            lo = min(lo, float(finite.min()))
            hi = max(hi, float(finite.max()))
    if lo > hi:
        return 0.0, 1.0
    return lo, hi


def replot_catalog(h5_path: str) -> list[ReplotGroup]:
    """List every aligned rocking product (3-D dataset) as a replot group."""
    groups: list[ReplotGroup] = []
    with h5py.File(h5_path, "r") as f:
        z_um = f["z_uniform_um"][:].tolist() if "z_uniform_um" in f else None
        for key, (title, _cbar) in _DATASET_DISPLAY.items():
            obj = f.get(key)
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                continue
            n_z = obj.shape[0]
            labels = [
                f"layer {z}" + (f"  (Z={z_um[z]:.2f} µm)" if z_um else "") for z in range(n_z)
            ]
            groups.append(ReplotGroup(key=key, label=title, item_labels=labels))
    return groups


def render_replot(h5_path, selections, style, clim, out_dir, roi=None, params=None) -> list[str]:
    """Re-render selected aligned rocking map layers cold from an aligned h5.

    ``selections`` is ``list[(dataset_key, item_idxs | None)]`` where dataset_key
    is ``sum_intensity`` or ``specific_frame``. ``clim`` overrides vmin/vmax;
    ``roi`` crops each layer. PNGs under ``{out_dir}/{key}/``; returns paths.
    """
    params = params or {}
    px = float(params.get("pixel_size_x_um", STAGE.defaults()["pixel_size_x_um"]))
    py = float(params.get("pixel_size_y_um", STAGE.defaults()["pixel_size_y_um"]))
    written: list[str] = []
    with h5py.File(h5_path, "r") as f:
        z_um = f["z_uniform_um"][:].tolist() if "z_uniform_um" in f else None
        for key, idxs in selections:
            obj = f.get(key)
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                continue
            title, cbar_label = _DATASET_DISPLAY.get(key, (key, "Intensity (a.u.)"))
            vmin, vmax = _layer_clim(obj)
            n_z = obj.shape[0]
            layer_list = list(range(n_z)) if idxs is None else list(idxs)
            sub_dir = os.path.join(out_dir, key)
            os.makedirs(sub_dir, exist_ok=True)
            for z in layer_list:
                if z < 0 or z >= n_z:
                    continue
                fig = render_volume_layer(
                    h5_path, key, z,
                    cmap="gray", cmap_group="raw", title=title, cbar_label=cbar_label,
                    sx=px, sy=py, vmin=vmin, vmax=vmax, style=style, clim=clim, roi=roi,
                    z_um=z_um,
                )
                if fig is None:
                    continue
                png = os.path.join(sub_dir, f"{key}_layer_{z:04d}.png")
                fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
                written.append(png)
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_rocking.py -q`
Expected: PASS.

- [ ] **Step 5: Update docs + commit**

Add rocking Replot to `docs/Usage.md` and the functions to `docs/Codebase.md`.

```bash
ruff check dfxm/stages/rocking.py tests/test_stage_rocking.py
git add dfxm/stages/rocking.py tests/test_stage_rocking.py docs/Usage.md docs/Codebase.md
git commit -m "feat(rocking): cold replot_catalog + render_replot with clim/ROI"
```

---

### Task 4: strain `replot_catalog` + `render_replot`

**Files:**
- Modify: `dfxm/stages/strain.py`
- Test: `tests/test_stage_strain.py`
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `ReplotGroup`, `crop_roi_2d` (Task 1); existing `build_strain_map`, `_parse_roi`, `_parse_float`.
- Produces:
  - `strain._rebuild_strain_map(h5_path, layer_idx, style, *, clim=None, roi=None, params=None) -> Figure` — reads `strain[layer_idx]`, optional crop, builds the RdBu_r map with a clim override.
  - `strain.replot_catalog(h5_path) -> list[ReplotGroup]` — single group `strain`, one item per layer (from `f.attrs["source_folders"]`).
  - `strain.render_replot(h5_path, selections, style, clim, out_dir, roi=None, params=None) -> list[str]` — `selections` is `list[(key, item_idxs | None)]` with the single key `"strain"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stage_strain.py`:

```python
def _write_strain_vol(path, names=("a", "b")):
    import h5py
    import numpy as np

    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as f:
        f.create_dataset("strain", data=rng.standard_normal((len(names), 4, 5)).astype(np.float32))
        f.attrs["num_layers"] = len(names)
        f.attrs["source_folders"] = "\n".join(names)
        f.attrs["scale_x_um"] = 0.152
        f.attrs["scale_y_um"] = 0.385
    return path


def test_strain_replot_catalog_single_group_per_layer(tmp_path):
    h5 = str(tmp_path / "strain.h5")
    _write_strain_vol(h5, names=("a", "b", "c"))
    cat = ST.replot_catalog(h5)
    assert len(cat) == 1 and cat[0].key == "strain"
    assert cat[0].item_labels == ["a", "b", "c"]


def test_strain_rebuild_map_clim_override(tmp_path):
    h5 = str(tmp_path / "strain.h5")
    _write_strain_vol(h5)
    fig = ST._rebuild_strain_map(h5, 0, style=None, clim=(-1e-3, 1e-3))
    im = fig.axes[0].images[0]
    assert im.norm.vmin == -1e-3 and im.norm.vmax == 1e-3


def test_strain_render_replot_writes_pngs_with_crop(tmp_path):
    import os

    h5 = str(tmp_path / "strain.h5")
    _write_strain_vol(h5, names=("a", "b"))
    out = str(tmp_path / "replots")
    written = ST.render_replot(
        h5, [("strain", [0])], style=None, clim=None, out_dir=out, roi=(0, 2, 0, 3)
    )
    assert len(written) == 1 and os.path.exists(written[0])
```

(Match the module alias the test file already uses for `strain`, e.g. `ST`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_strain.py -k replot -v`
Expected: FAIL with `AttributeError: ... 'strain' has no attribute 'replot_catalog'`.

- [ ] **Step 3: Implement**

In `dfxm/stages/strain.py` add (import `from dfxm.common.figures import ReplotGroup, crop_roi_2d`):

```python
def _rebuild_strain_map(h5_path, layer_idx, style, *, clim=None, roi=None, params=None):
    """Rebuild one strain-map Figure from the stacked volume, cold, with clim/ROI.

    ``roi`` (pixel bounds) crops the stored layer; the cropped view uses a
    zero-origin extent. ``clim`` overrides the symmetric vlim.
    """
    params = params or {}
    with h5py.File(h5_path, "r") as fh:
        arr = fh["strain"][layer_idx]
        px = float(params.get("pixel_size_x_um") or fh.attrs.get("scale_x_um", 0.152))
        py = float(params.get("pixel_size_y_um") or fh.attrs.get("scale_y_um", 0.385))
    if roi is not None:
        cropped = crop_roi_2d(arr, roi)
        if cropped is None:
            return None
        arr = cropped
        extent_roi = None  # cropped view → zero-origin extent
    else:
        extent_roi = _parse_roi(str(params.get("roi", "") or ""))
    if clim is not None:
        vlim = clim
    else:
        vlim = (_parse_float(params.get("vmin", "")), _parse_float(params.get("vmax", "")))
    return build_strain_map(arr, px, py, extent_roi, vlim, style=style)


def replot_catalog(h5_path: str) -> list[ReplotGroup]:
    """Single 'strain' group; one item per stored layer (source-folder names)."""
    with h5py.File(h5_path, "r") as f:
        raw = str(f.attrs.get("source_folders", ""))
        n = int(f.attrs.get("num_layers", f["strain"].shape[0]))
    names = [s for s in raw.split("\n") if s] if raw else [f"layer {i}" for i in range(n)]
    if len(names) != n:
        names = [f"layer {i}" for i in range(n)]
    return [ReplotGroup(key="strain", label="Strain map", item_labels=names)]


def render_replot(h5_path, selections, style, clim, out_dir, roi=None, params=None) -> list[str]:
    """Re-render selected strain map layers cold from the stacked strain h5.

    ``selections`` is ``list[("strain", item_idxs | None)]``. PNGs under
    ``{out_dir}/strain/``; returns written paths.
    """
    with h5py.File(h5_path, "r") as f:
        n_z = int(f["strain"].shape[0])
        names = str(f.attrs.get("source_folders", "")).split("\n")
    sub_dir = os.path.join(out_dir, "strain")
    os.makedirs(sub_dir, exist_ok=True)
    written: list[str] = []
    for key, idxs in selections:
        if key != "strain":
            continue
        layer_list = list(range(n_z)) if idxs is None else list(idxs)
        for z in layer_list:
            if z < 0 or z >= n_z:
                continue
            fig = _rebuild_strain_map(h5_path, z, style, clim=clim, roi=roi, params=params)
            if fig is None:
                continue
            stem = names[z] if z < len(names) and names[z] else f"layer_{z:04d}"
            png = os.path.join(sub_dir, f"{stem}_strain.png")
            fig.savefig(png, dpi=200, facecolor="white", bbox_inches="tight")
            written.append(png)
    return written
```

Note: confirm `_parse_roi` / `_parse_float` accept the string/blank inputs used here (they are the same helpers `figures()` uses — reuse verbatim; if `_parse_float` expects a specific blank sentinel, match `figures()`'s call at strain.py:556).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_strain.py -q`
Expected: PASS.

- [ ] **Step 5: Update docs + commit**

Add strain Replot to `docs/Usage.md` (note the ROI-crop-only-within-run-ROI + cold-axis caveats) and the functions to `docs/Codebase.md`.

```bash
ruff check dfxm/stages/strain.py tests/test_stage_strain.py
git add dfxm/stages/strain.py tests/test_stage_strain.py docs/Usage.md docs/Codebase.md
git commit -m "feat(strain): cold replot_catalog + render_replot with clim/ROI"
```

---

### Task 5: slices — add `roi` to `render_replot` + `_rebuild_plane_figure`

**Files:**
- Modify: `dfxm/stages/slices.py`
- Test: `tests/test_stage_slices.py`
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `crop_roi_2d` (Task 1); existing `_rebuild_plane_figure`, `render_replot`, `build_slice_figure`.
- Produces (signature changes, backward-compatible defaults):
  - `slices._rebuild_plane_figure(h5_path, vid, sname, k, style, *, clim=None, roi=None) -> Figure | None`
  - `slices.render_replot(h5_path, selections, style, clim, out_dir, roi=None, *, dpi=150) -> list[str]`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_slices.py`:

```python
def test_render_replot_roi_crops_slice(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    # crop to a sub-window; the rebuilt image must have the cropped shape
    fig = S._rebuild_plane_figure(str(h5), "strain", "plane_a", 1, style=None, roi=(0, 2, 0, 2))
    im = fig.axes[0].images[0]
    assert im.get_array().shape == (2, 2)


def test_render_replot_roi_empty_crop_skipped(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    written = S.render_replot(
        str(h5), [("strain", "plane_a", None)], style=None, clim=None,
        out_dir=str(tmp_path / "r"), roi=(2, 2, 0, 3),
    )
    assert written == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_slices.py -k "roi" -v`
Expected: FAIL with `TypeError: _rebuild_plane_figure() got an unexpected keyword argument 'roi'`.

- [ ] **Step 3: Implement**

In `dfxm/stages/slices.py`, import `from dfxm.common.figures import crop_roi_2d`. Change `_rebuild_plane_figure` to accept `roi` and crop `s2d` + `u`/`v` (u = columns/x, v = rows/y):

```python
def _rebuild_plane_figure(h5_path, vid, sname, k, style, *, clim=None, roi=None):
    with h5py.File(h5_path, "r") as f:
        vg = f[vid]
        kind = str(vg.attrs.get("kind", ""))
        prep = {
            "cmap_name": str(vg.attrs.get("cmap", "magma")),
            "title": str(vg.attrs.get("title", vid)),
            "cbar_label": str(vg.attrs.get("cbar_label", "")),
            "vmin": float(vg.attrs.get("vmin", 0.0)),
            "vmax": float(vg.attrs.get("vmax", 1.0)),
            "center_zero": kind in _CENTERED_KINDS,
        }
        sg = vg[sname]
        s2d = sg["slices"][k]
        u = sg["u_um"][:]
        v = sg["v_um"][:]
        off = float(sg["offsets_um"][k])
    if roi is not None:
        cropped = crop_roi_2d(s2d, roi)
        if cropped is None:
            return None
        r0, r1, c0, c1 = roi
        h, w = s2d.shape[:2]
        r0 = max(0, min(int(r0), h)); r1 = max(0, min(int(r1), h))
        c0 = max(0, min(int(c0), w)); c1 = max(0, min(int(c1), w))
        s2d, u, v = cropped, u[c0:c1], v[r0:r1]
    if clim is not None:
        vmin_o, vmax_o = clim
        if vmin_o is not None:
            prep["vmin"] = float(vmin_o)
        if vmax_o is not None:
            prep["vmax"] = float(vmax_o)
    prep["cmap_name"] = resolve_cmap(style, GROUP_BY_KIND.get(kind), fallback=prep["cmap_name"])
    prep["group"] = GROUP_BY_KIND.get(kind)
    return build_slice_figure(prep, {"name": sname}, s2d, u, v, offset_um=off, style=style)
```

Then thread `roi` through `render_replot` and skip `None` figures:

```python
def render_replot(h5_path, selections, style, clim, out_dir, roi=None, *, dpi=150) -> list[str]:
    catalog = {(e.volume_id, e.slice_name): e for e in replot_catalog(h5_path)}
    written: list[str] = []
    for vid, sname, plane_idxs in selections:
        entry = catalog.get((vid, sname))
        if entry is None:
            continue
        idxs = list(range(entry.n_planes)) if plane_idxs is None else list(plane_idxs)
        slice_dir = os.path.join(out_dir, sname)
        os.makedirs(slice_dir, exist_ok=True)
        for k in idxs:
            if k < 0 or k >= entry.n_planes:
                continue
            fig = _rebuild_plane_figure(h5_path, vid, sname, k, style, clim=clim, roi=roi)
            if fig is None:
                continue
            if entry.n_planes == 1:
                png = os.path.join(slice_dir, f"{vid}.png")
            else:
                png = os.path.join(slice_dir, f"{vid}__p{k:03d}_{entry.offsets_um[k]:+08.2f}um.png")
            fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight")
            written.append(png)
    return written
```

Note: `figures()` also calls `_rebuild_plane_figure` (its `build` closure) — the new keyword-only args default to `None`, so that call is unchanged. Verify the call site (slices.py ~line 1050) still passes only positional `style`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_slices.py -q`
Expected: PASS (including the pre-existing replot + figures tests).

- [ ] **Step 5: Update docs + commit**

Note the new ROI crop on slices in `docs/Usage.md`; update `_rebuild_plane_figure`/`render_replot` signatures in `docs/Codebase.md`.

```bash
ruff check dfxm/stages/slices.py tests/test_stage_slices.py
git add dfxm/stages/slices.py tests/test_stage_slices.py docs/Usage.md docs/Codebase.md
git commit -m "feat(slices): ROI crop override in replot (render_replot/_rebuild_plane_figure)"
```

---

### Task 6: generic `ReplotDialog` (GUI)

**Files:**
- Create: `gui/widgets/replot_dialog.py`
- Test: `tests/test_gui_replot_dialog.py` (create)
- Docs: `docs/Codebase.md`

**Interfaces:**
- Consumes: `dfxm.common.figures.ReplotGroup`.
- Produces:
  - `class ReplotDialog(QDialog)` with `__init__(self, h5_default, catalog_fn, render_fn, style=None, out_default="", parent=None)` where `catalog_fn(h5_path) -> list[ReplotGroup]` and `render_fn(h5_path, selections, style, clim, roi, out_dir) -> list[str]`; `selections` is `list[(group_key, item_idxs | None)]`. A file field (editable + Browse + Load) reloads the catalog via `catalog_fn`, mirroring `SliceReplotDialog` — this keeps cold-start (open on a file the current session never ran).
  - `.written: list[str]`, `.select_all()`, `.render_selection(out_dir) -> list[str]`.
- **ROI contract (shared with Task 7):** all four boxes must be filled together to take effect; any blank → ROI ignored (`None`). `crop_roi_2d` needs four ints, so partial ROI is never forwarded.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_replot_dialog.py` (headless Qt, mirroring `test_gui_slice_replot.py`):

```python
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from dfxm.common.figures import ReplotGroup  # noqa: E402
from gui.widgets.replot_dialog import ReplotDialog  # noqa: E402

_app = QApplication.instance() or QApplication([])


def test_replot_dialog_collects_selection_clim_roi(tmp_path):
    h5 = tmp_path / "vol.h5"
    h5.write_bytes(b"")  # existence is all the dialog checks before catalog_fn
    captured = {}

    def catalog_fn(path):
        return [ReplotGroup(key="A", label="A", item_labels=["l0", "l1"])]

    def render_fn(path, selections, style, clim, roi, out_dir):
        captured["selections"] = selections
        captured["clim"] = clim
        captured["roi"] = roi
        return [os.path.join(out_dir, "x.png")]

    dlg = ReplotDialog(str(h5), catalog_fn, render_fn, style=None, out_default=str(tmp_path))
    dlg.select_all()
    dlg._vmin.setText("0.5")
    dlg._r0.setText("0")
    dlg._r1.setText("2")
    dlg._c0.setText("0")
    dlg._c1.setText("3")
    written = dlg.render_selection(str(tmp_path))
    assert captured["selections"] == [("A", None)]
    assert captured["clim"] == (0.5, None)
    assert captured["roi"] == (0, 2, 0, 3)
    assert written == [os.path.join(str(tmp_path), "x.png")]


def test_replot_dialog_partial_roi_ignored(tmp_path):
    h5 = tmp_path / "vol.h5"
    h5.write_bytes(b"")
    captured = {}

    def render_fn(path, selections, style, clim, roi, out_dir):
        captured["roi"] = roi
        return []

    dlg = ReplotDialog(
        str(h5),
        lambda p: [ReplotGroup(key="A", label="A", item_labels=["l0"])],
        render_fn,
        out_default=str(tmp_path),
    )
    dlg.select_all()
    dlg._r0.setText("0")  # only one box filled → ROI ignored
    dlg.render_selection(str(tmp_path))
    assert captured["roi"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_gui_replot_dialog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gui.widgets.replot_dialog'`.

- [ ] **Step 3: Implement**

Create `gui/widgets/replot_dialog.py`. A file field (Browse/Load) + group→item checkable tree (2-level, tristate) + clim boxes + ROI boxes (`r0,r1,c0,c1`) + output dir + Render/Close. **ROI contract: all four boxes filled → `(r0,r1,c0,c1)`; otherwise `None`.** `render_fn` is called with `(h5_path, selections, style, clim, roi, out_dir)`.

```python
"""Generic 2-level replot dialog (group → layer), built lazily on demand.

Consumes a catalog_fn + render callback from the Qt-free core, so it serves
strain/mosaicity/rocking uniformly. clim + pixel-index ROI crop override the
stored render; nothing is recomputed. The file field keeps cold-start working.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)


class ReplotDialog(QDialog):
    """Pick groups/layers from a replot catalog (via catalog_fn) and re-render PNGs."""

    def __init__(
        self, h5_default, catalog_fn, render_fn, style=None, out_default="", parent=None
    ) -> None:
        super().__init__(parent)
        self._h5_path = h5_default or ""
        self._catalog_fn = catalog_fn
        self._render_fn = render_fn
        self._style = style
        self.written: list[str] = []
        self.setWindowTitle("Replot")

        self._file_edit = QLineEdit(self._h5_path)
        file_browse = QPushButton("Browse…")
        file_browse.clicked.connect(self._on_browse_h5)
        file_reload = QPushButton("Load")
        file_reload.clicked.connect(self._reload)
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Volume file:"))
        file_row.addWidget(self._file_edit, 1)
        file_row.addWidget(file_browse)
        file_row.addWidget(file_reload)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Group / layer"])
        select_all_btn = QPushButton("Select all")
        select_all_btn.clicked.connect(self.select_all)
        deselect_btn = QPushButton("Deselect all")
        deselect_btn.clicked.connect(self._deselect_all)
        toolbar = QHBoxLayout()
        toolbar.addWidget(select_all_btn)
        toolbar.addWidget(deselect_btn)
        toolbar.addStretch(1)

        self._vmin = QLineEdit()
        self._vmin.setPlaceholderText("vmin")
        self._vmax = QLineEdit()
        self._vmax.setPlaceholderText("vmax")
        clim_row = QHBoxLayout()
        clim_row.addWidget(QLabel("Colour limits:"))
        clim_row.addWidget(self._vmin)
        clim_row.addWidget(self._vmax)

        self._r0, self._r1 = QLineEdit(), QLineEdit()
        self._c0, self._c1 = QLineEdit(), QLineEdit()
        for e, ph in ((self._r0, "r0"), (self._r1, "r1"), (self._c0, "c0"), (self._c1, "c1")):
            e.setPlaceholderText(ph)
        roi_row = QHBoxLayout()
        roi_row.addWidget(QLabel("ROI crop (px, blank=full):"))
        for e in (self._r0, self._r1, self._c0, self._c1):
            roi_row.addWidget(e)

        self._out_edit = QLineEdit(out_default)
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._on_browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output dir:"))
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(out_browse)

        self._status = QLabel("")
        render_btn = QPushButton("Render")
        render_btn.setProperty("role", "primary")
        render_btn.clicked.connect(self._on_render)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._status, 1)
        btn_row.addWidget(render_btn)
        btn_row.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(file_row)
        layout.addWidget(self._tree, 1)
        layout.addLayout(toolbar)
        layout.addLayout(clim_row)
        layout.addLayout(roi_row)
        layout.addLayout(out_row)
        layout.addLayout(btn_row)
        self._reload()

    def _reload(self) -> None:
        self._h5_path = self._file_edit.text().strip()
        self._tree.clear()
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._status.setText("no such file")
            return
        try:
            catalog = self._catalog_fn(self._h5_path)
        except Exception as exc:  # noqa: BLE001 — GUI reload: show status, never crash
            self._status.setText(f"cannot read: {exc}")
            return
        for grp in catalog:
            top = QTreeWidgetItem(self._tree, [grp.label])
            top.setFlags(top.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
            top.setCheckState(0, Qt.CheckState.Unchecked)
            top.setData(0, Qt.ItemDataRole.UserRole, grp.key)
            for z, lab in enumerate(grp.item_labels):
                leaf = QTreeWidgetItem(top, [lab])
                leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setCheckState(0, Qt.CheckState.Unchecked)
                leaf.setData(0, Qt.ItemDataRole.UserRole, z)
        self._tree.expandAll()
        self._status.setText(f"{self._tree.topLevelItemCount()} group(s)")

    def select_all(self) -> None:
        for i in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Checked)

    def _deselect_all(self) -> None:
        for i in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Unchecked)

    def _selections(self):
        sels = []
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            key = top.data(0, Qt.ItemDataRole.UserRole)
            checked = [
                top.child(k).data(0, Qt.ItemDataRole.UserRole)
                for k in range(top.childCount())
                if top.child(k).checkState(0) == Qt.CheckState.Checked
            ]
            if top.checkState(0) == Qt.CheckState.Checked and len(checked) == top.childCount():
                sels.append((key, None))  # whole group = all layers
            elif checked:
                sels.append((key, checked))
        return sels

    @staticmethod
    def _f(edit):
        t = edit.text().strip()
        return float(t) if t else None

    @staticmethod
    def _i(edit):
        t = edit.text().strip()
        return int(t) if t else None

    def _clim(self):
        vmin, vmax = self._f(self._vmin), self._f(self._vmax)
        return None if (vmin is None and vmax is None) else (vmin, vmax)

    def _roi(self):
        vals = (self._i(self._r0), self._i(self._r1), self._i(self._c0), self._i(self._c1))
        return vals if all(v is not None for v in vals) else None  # all-four-or-none

    def render_selection(self, out_dir):
        self.written = self._render_fn(
            self._h5_path, self._selections(), self._style, self._clim(), self._roi(), out_dir
        )
        return self.written

    def _on_render(self) -> None:
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            self._status.setText("set an output dir")
            return
        if not self._selections():
            self._status.setText("nothing selected")
            return
        try:
            written = self.render_selection(out_dir)
        except Exception as exc:  # noqa: BLE001 — surface render errors in the status bar
            self._status.setText(f"render failed: {exc}")
            return
        self._status.setText(f"wrote {len(written)} PNG(s) → {out_dir}")

    def _on_browse_h5(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open volume h5", "", "HDF5 (*.h5)")
        if path:
            self._file_edit.setText(path)
            self._reload()

    def _on_browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output directory")
        if path:
            self._out_edit.setText(path)
```

The `_roi()` all-four-or-none contract (partial ROI → `None`) matches `crop_roi_2d`, which needs four ints, and keeps parity with `SliceReplotDialog` (Task 7). The Step-1 tests assert both branches.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_gui_replot_dialog.py -v`
Expected: PASS.

- [ ] **Step 5: Update docs + commit**

Add `ReplotDialog` to `docs/Codebase.md` (`gui/widgets`).

```bash
ruff check gui/widgets/replot_dialog.py tests/test_gui_replot_dialog.py
git add gui/widgets/replot_dialog.py tests/test_gui_replot_dialog.py docs/Codebase.md
git commit -m "feat(gui): generic ReplotDialog (group/layer tree + clim + ROI)"
```

---

### Task 7: ROI boxes on `SliceReplotDialog`

**Files:**
- Modify: `gui/widgets/slice_replot.py`
- Test: `tests/test_gui_slice_replot.py`
- Docs: `docs/Codebase.md`

**Interfaces:**
- Consumes: `slices.render_replot(..., roi=...)` (Task 5).
- Produces: `SliceReplotDialog` gains `_r0/_r1/_c0/_c1` boxes + `_roi()`; `render_selection` passes `roi`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gui_slice_replot.py`:

```python
def test_slice_replot_dialog_passes_roi(tmp_path, monkeypatch):
    from dfxm.stages import slices as sl

    captured = {}

    def fake_render_replot(h5, selections, style, clim, out_dir, roi=None, **kw):
        captured["roi"] = roi
        return []

    monkeypatch.setattr(sl, "render_replot", fake_render_replot)
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))  # reuse the module's fixture helper
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path))
    dlg.select_all()
    dlg._r0.setText("0")
    dlg._r1.setText("2")
    dlg._c0.setText("0")
    dlg._c1.setText("2")
    dlg.render_selection(str(tmp_path))
    assert captured["roi"] == (0, 2, 0, 2)
```

(If `_write_mini_consolidated` lives in `test_stage_slices.py`, import it or add a small local h5 writer.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_gui_slice_replot.py -k roi -v`
Expected: FAIL with `AttributeError: 'SliceReplotDialog' object has no attribute '_r0'`.

- [ ] **Step 3: Implement**

In `gui/widgets/slice_replot.py`, add ROI boxes (parallel to the clim boxes at ~line 68-76) and a `_roi()` parser, and pass `roi` in `render_selection` (line ~196). Add after the clim row:

```python
        self._r0, self._r1 = QLineEdit(), QLineEdit()
        self._c0, self._c1 = QLineEdit(), QLineEdit()
        for e, ph in ((self._r0, "r0"), (self._r1, "r1"), (self._c0, "c0"), (self._c1, "c1")):
            e.setPlaceholderText(ph)
        roi_row = QHBoxLayout()
        roi_row.addWidget(QLabel("ROI crop (px, blank=full):"))
        for e in (self._r0, self._r1, self._c0, self._c1):
            roi_row.addWidget(e)
```

Insert `layout.addLayout(roi_row)` right after `layout.addLayout(clim_row)`. Add:

```python
    def _roi(self):
        def _i(edit):
            t = edit.text().strip()
            return int(t) if t else None

        vals = (_i(self._r0), _i(self._r1), _i(self._c0), _i(self._c1))
        if all(v is None for v in vals):
            return None
        if any(v is None for v in vals):
            return None  # partial ROI ignored; keep parity with the four-box contract
        return vals
```

Change `render_selection` to pass `roi`:

```python
    def render_selection(self, out_dir):
        self.written = _sl.render_replot(
            self._h5_path, self._selections(), self._style, self._clim(), out_dir, roi=self._roi()
        )
        return self.written
```

(Keep the four-box contract identical to Task 6 so both dialogs behave the same.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_gui_slice_replot.py -q`
Expected: PASS.

- [ ] **Step 5: Update docs + commit**

Note the slices ROI boxes in `docs/Usage.md` and the `_roi` addition in `docs/Codebase.md`.

```bash
ruff check gui/widgets/slice_replot.py tests/test_gui_slice_replot.py
git add gui/widgets/slice_replot.py tests/test_gui_slice_replot.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): ROI crop boxes on SliceReplotDialog"
```

---

### Task 8: un-gate `Replot…` button + per-stage dispatch in `stage_view.py`

**Files:**
- Modify: `gui/stage_view.py`
- Test: `tests/test_stage_view_buttons.py` (create) + `tests/gui_smoke.py`
- Docs: `docs/Usage.md`

**Interfaces:**
- Consumes: `<stage>.replot_catalog` (passed as `catalog_fn`) / `<stage>.render_replot` (Tasks 2-4), `ReplotDialog(h5_default, catalog_fn, render_fn, …)` (Task 6), existing `SliceReplotDialog`; `self._last_result` (`.stacked_path` / `.aligned_path`) for the default h5.
- Produces: `Replot…` button present for `{slices, strain, mosaicity, rocking}`; `_on_replot` dispatches per stage (`_replot_slices` for slices, generic `ReplotDialog` otherwise).

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage_view_buttons.py`:

```python
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from dfxm.config.experiment import Experiment  # noqa: E402  (match the real import path)
from gui.bindings import STAGE_SPECS  # noqa: E402
from gui.stage_view import StageView  # noqa: E402

_app = QApplication.instance() or QApplication([])

_REPLOT_STAGES = {"slices", "strain", "mosaicity", "rocking"}


@pytest.mark.parametrize("stage", sorted(STAGE_SPECS))
def test_replot_button_only_on_map_stages(stage):
    view = StageView(stage, STAGE_SPECS[stage], Experiment())
    has_button = view._replot_btn is not None
    assert has_button == (stage in _REPLOT_STAGES)
```

(Match `Experiment()` construction / `StageView` constructor args to the real signatures used elsewhere in `tests/` — grep `StageView(` in existing GUI tests.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_view_buttons.py -v`
Expected: FAIL — strain/mosaicity/rocking currently have `_replot_btn is None`.

- [ ] **Step 3: Implement**

In `gui/stage_view.py`, change the button gate (line ~104) from `if stage_name == "slices":` to:

```python
        self._replot_btn: QPushButton | None = None
        if stage_name in ("slices", "strain", "mosaicity", "rocking"):
            self._replot_btn = QPushButton("Replot…")
            self._replot_btn.clicked.connect(self._on_replot)
            btn_row.addWidget(self._replot_btn)
```

Refactor `_on_replot` (line ~328) to keep the slices path and add the generic path. The generic dialog carries its own file field, so `_on_replot` only supplies a **best-effort default h5** — the exact path the last run wrote (`self._last_result.stacked_path` / `.aligned_path`) — and never needs to reconstruct the mode-dependent output path:

```python
    def _on_replot(self) -> None:
        import time
        from dataclasses import replace

        vals = self._form.values()
        window = self.window()
        style = window.global_plot_style() if hasattr(window, "global_plot_style") else None
        style = replace(style) if style is not None else None
        ts = time.strftime("%Y%m%d-%H%M%S")

        if self._stage_name == "slices":
            self._replot_slices(vals, style, ts)
            return

        from dfxm.stages import mosaicity as _mo, rocking as _ro, strain as _st

        module = {"strain": _st, "mosaicity": _mo, "rocking": _ro}[self._stage_name]
        # Best-effort default h5 = the exact path the last run wrote (if any); the
        # dialog's file field lets the user Browse/Load a different one (cold start).
        res = self._last_result
        h5_default = ""
        for attr in ("stacked_path", "aligned_path"):
            p = getattr(res, attr, "") if res is not None else ""
            if p:
                h5_default = p
                break
        base = os.path.dirname(h5_default) if h5_default else "."
        out_dir = os.path.join(base, "replots", ts)

        from .widgets.replot_dialog import ReplotDialog  # imported on demand

        def render_fn(h5, selections, st, clim, roi, out, _m=module, _p=dict(vals)):
            return _m.render_replot(h5, selections, st, clim, out, roi=roi, params=_p)

        dlg = ReplotDialog(
            h5_default,
            module.replot_catalog,
            render_fn,
            style=style,
            out_default=out_dir,
            parent=self,
        )
        dlg.exec()
        if dlg.written:
            self._log.append(
                f"Replotted {len(dlg.written)} PNG(s) → {os.path.dirname(dlg.written[0])}"
            )
            self._tabs.setCurrentWidget(self._log)
```

Move the existing slices body into a `_replot_slices(self, vals, style, ts)` helper (verbatim from the current `_on_replot`, minus the `style`/`window` lines now hoisted above; reuse the passed `ts` for its `replots/<ts>/` dir).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_view_buttons.py -q && python3 tests/gui_smoke.py`
Expected: PASS; smoke run prints its `[N]` checks with no error.

- [ ] **Step 5: Update docs + commit**

Update `docs/Usage.md` Replot section to say the button is available on slices + strain + mosaicity + rocking, with the one-clim-per-batch + ROI-crop-bounded-by-stored-data caveats.

```bash
ruff check gui/stage_view.py tests/test_stage_view_buttons.py
git add gui/stage_view.py tests/test_stage_view_buttons.py docs/Usage.md
git commit -m "feat(gui): Replot… button + per-stage dispatch for strain/mosaicity/rocking"
```

---

## Final verification

- [ ] Run the full suite: `python3 -m pytest -q` — expected all pass (prior baseline 330 passed / 13 skipped + the new tests), zero warnings.
- [ ] `ruff check . && ruff format --check .`
- [ ] `python3 tests/gui_smoke.py` — all `[N]` checks pass.
- [ ] Sanity: launch `python3 -m gui.app`, run (or point at an existing) strain/mosaicity/rocking volume, click **Replot…**, pick a layer subset, set a clim + an ROI crop, Render, and confirm PNGs land under `…/replots/<ts>/`.

## Notes for the implementer

- **Default h5 (Task 8) comes from `self._last_result`, not form-key reconstruction:** the run result carries the exact output path (`stacked_path` for strain/mosaicity, `aligned_path` for rocking). The dialog's file field covers cold-start (no run this session) — the user Browses/Loads. Do **not** try to re-derive each stage's mode-dependent `default_out_root` in the GUI.
- **Module aliases in tests:** `tests/test_stage_*.py` import the stage under a short alias (`M`, `R`, `ST`/`S`). Match the file's existing alias; don't introduce a new one, and confirm it before pasting the test bodies.
- **`_streamed_clim` is mosaicity-local:** rocking gets its own `_layer_clim` (Task 3) rather than importing a private helper across stages.
- **`_write_mini_consolidated`** (used in Task 5/7 tests) already exists in `tests/test_stage_slices.py`; import it or add a minimal local writer in the GUI test.
- **StageView constructor args (Task 8 test):** grep an existing GUI test for `StageView(` and the real `Experiment` import path before writing `test_stage_view_buttons.py`.
