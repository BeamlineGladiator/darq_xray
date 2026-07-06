# Slices replot + mosa-sum raw source + per-job profiles — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an appearance-only slices "Replot…" subwidget (rebuilds PNGs from `oblique_slices.h5` with no resampling), a mosa-scan sum-intensity raw volume as a new parallel source on the rocking stage, slices/profiles consumption of it, per-direction PNG subfolders, and per-job field selection in line profiles.

**Architecture:** All heavy logic stays in the Qt-free `dfxm/` core (new `slices.replot_catalog` / `slices.render_replot` + a shared plane-rebuild helper); the GUI dialogs are thin shells that call it. The mosa-sum source is a `source_scan` branch on the existing rocking stage that reuses its frame-read / alignment / save / render machinery. Per-job profile field selection is a two-line change inside `profiles._collect` that both `run()` and the figure catalog inherit.

**Tech Stack:** Python 3.10, numpy, h5py, scipy (`map_coordinates`), matplotlib (explicit `Figure` API), PySide6 (GUI only), pytest.

## Global Constraints

- **Keep `dfxm/` Qt-free** — never import PySide6/pyvista/vtk under `dfxm/`. GUI code lives only under `gui/`.
- **Plotting via `matplotlib.figure.Figure`** — never `pyplot` or `matplotlib.use(...)`.
- **One alignment** — reuse `dfxm/common/alignment.py`; never reimplement samy-shift / Z-interpolation.
- **Every new `Param` needs `help`**; advanced params need `group`; input paths set `must_exist=True`. `tests/test_param_metadata.py` enforces this.
- **Raise `StageUserError(message, hint=...)`** (from `dfxm.common.errors`) for user-fixable input problems; keep skip-based reporting (result lists) otherwise.
- **Docs contract:** every task that changes a stage's params/behaviour/IO or a viewer updates BOTH `docs/Usage.md` (user-facing) and `docs/Codebase.md` (code reference) in the SAME commit.
- **ruff**: line length 100, double quotes, target py310, rules E/F/I. `ruff format` runs automatically on Write/Edit via the settings hook.
- **No git remote** — never pull/push. Commit locally only.
- **Commit trailers:** every commit message ends with these two lines:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125LtLRuM6NRqV3rxx57b2S
  ```
  (Abbreviated as `<trailers>` in the commit steps below — always include them.)
- **Run the suite** with `python3 -m pytest -q`; lint with `ruff check .`.

## File Structure

- `dfxm/stages/slices.py` — Task 1 (per-direction PNG subfolders in `run()`), Task 4 (mosa file params + `raw_mosa_*` kinds), Task 5 (`replot_catalog`, `render_replot`, shared `_rebuild_plane_figure`).
- `dfxm/common/plotting.py` — Task 4 (`GROUP_BY_KIND` additions).
- `dfxm/stages/rocking.py` — Task 2 (`subtract_background` through `process_raw_scan`), Task 3 (`source_scan` mosaicity branch + output naming + source-aware titles).
- `dfxm/stages/profiles.py` — Task 7 (per-job `fields`/`reference` in `_collect`).
- `gui/widgets/slice_replot.py` — Task 6 (new `SliceReplotDialog`).
- `gui/widgets/line_picker.py` — Task 8 (field checkboxes).
- `gui/stage_view.py` — Task 6 ("Replot…" button + slot), Task 8 (write `fields` into the injected job).
- `gui/bindings.py` — Task 4 (`aligned_mosa_file` experiment-override default + `_ALIGNED_MOSA` constant).
- Tests: `tests/test_stage_slices.py`, `tests/test_stage_rocking.py`, `tests/test_stage_profiles.py`, `tests/test_gui_slice_replot.py` (new), `tests/test_gui_line_picker_fields.py` (new).
- Docs: `docs/Usage.md`, `docs/Codebase.md`.

---

## Task 1: Slices — per-direction PNG subfolders

**Files:**
- Modify: `dfxm/stages/slices.py` (the `save_png` block inside `run()`, ~lines 934–948)
- Test: `tests/test_stage_slices.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: per-plane PNGs at `{out_dir}/{slice_name}/{volume_id}[__pNNN_{off}um].png`; `SlicesResult.pngs` still holds full paths.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_slices.py` (reuses the existing `_setup` fixture):

```python
def test_run_writes_pngs_under_per_slice_subfolders(tmp_path):
    proc, raw = _setup(tmp_path)
    out = tmp_path / "sl"
    slices_json = (
        '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
        '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
    )
    res = S.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "slices_json": slices_json,
            "output_dir": str(out),
        }
    )
    assert res.pngs and all(os.path.exists(p) for p in res.pngs)
    # every PNG lives under {out_dir}/mid/, not flat in {out_dir}
    for p in res.pngs:
        assert os.path.basename(os.path.dirname(p)) == "mid"
        assert not os.path.exists(os.path.join(str(out), os.path.basename(p)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_slices.py::test_run_writes_pngs_under_per_slice_subfolders -v`
Expected: FAIL (PNGs are written flat as `{out}/mid__strain.png`, so `dirname` basename is `sl`, not `mid`).

- [ ] **Step 3: Write minimal implementation**

In `dfxm/stages/slices.py`, replace the `if save_png:` block inside `run()` with a per-slice subfolder:

```python
                    if save_png:
                        slice_dir = os.path.join(out_dir, sl["name"])
                        os.makedirs(slice_dir, exist_ok=True)
                        if len(offsets) == 1:
                            png = os.path.join(slice_dir, f"{prep['volume_id']}.png")
                            save_slice_png(
                                prep, sl, s2d, u_um, v_um, png, offset_um=None, style=style
                            )
                        else:
                            png = os.path.join(
                                slice_dir,
                                f"{prep['volume_id']}__p{pi:03d}_{off:+08.2f}um.png",
                            )
                            save_slice_png(
                                prep, sl, s2d, u_um, v_um, png, offset_um=off, style=style
                            )
                        result.pngs.append(png)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_stage_slices.py -q`
Expected: PASS (all slices tests, including the existing `test_run_writes_consolidated_h5_and_pngs`).

- [ ] **Step 5: Update docs**

In `docs/Usage.md`, in the Oblique-slices stage section, note that per-plane PNGs are now written into one subfolder per slice direction (`<output_dir>/<slice name>/…`). In `docs/Codebase.md`, update the `slices.run()` description to mention the per-slice PNG subfolder layout.

- [ ] **Step 6: Commit**

```bash
git add dfxm/stages/slices.py tests/test_stage_slices.py docs/Usage.md docs/Codebase.md
git commit -m "feat(slices): write per-plane PNGs into one subfolder per slice direction

<trailers>"
```

---

## Task 2: Rocking — `subtract_background` toggle through `process_raw_scan`

**Files:**
- Modify: `dfxm/stages/rocking.py` (`process_raw_scan`, `build_raw_volumes`, `STAGE` params, `run()` call site)
- Test: `tests/test_stage_rocking.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `process_raw_scan(..., normalize_sum, subtract_background=True)` and `build_raw_volumes(..., normalize_sum, subtract_background, progress=...)`. When `subtract_background=False`, `sum_2d` is the plain frame sum and `specific_2d` is the raw specific frame (no median subtracted).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_rocking.py`:

```python
def test_process_raw_scan_no_background_subtraction(tmp_path):
    """subtract_background=False -> plain sum and raw specific frame (no median removed)."""
    frames = _rng_frames(3)
    folder = _write_motor_folder(str(tmp_path), "rock__1", 0.0, 0.0, frames=frames)
    h5p = os.path.join(folder, "rock__1.h5")
    sum_2d, spec_2d, n_frames, idx = RK.process_raw_scan(
        h5p, "1.1/measurement/pco_ff", None, None, None, normalize_sum=False,
        subtract_background=False,
    )
    np.testing.assert_allclose(sum_2d, frames.sum(axis=0), rtol=1e-5)
    np.testing.assert_allclose(spec_2d, frames[idx], rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_rocking.py::test_process_raw_scan_no_background_subtraction -v`
Expected: FAIL with `TypeError: process_raw_scan() got an unexpected keyword argument 'subtract_background'`.

- [ ] **Step 3: Write minimal implementation**

In `dfxm/stages/rocking.py`, extend `process_raw_scan` (add the param + branch the median subtraction):

```python
def process_raw_scan(
    h5_path: str,
    detector_path: str,
    roi_x: tuple | None,
    roi_y: tuple | None,
    specific_frame_idx: int | None,
    normalize_sum: bool,
    subtract_background: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Read one scan; return (sum_2d, specific_2d, n_frames, idx).

    With ``subtract_background`` (default) a per-pixel median across the scan's
    frames is removed before summing (rocking behaviour); otherwise a plain sum
    and the raw specific frame are returned (mosa-topograph behaviour).
    """
    with h5py.File(h5_path, "r") as f:
        det = f[detector_path]
        n_frames = det.shape[0]
        h_full, w_full = det.shape[1], det.shape[2]
        ys = roi_y[0] if roi_y else 0
        ye = roi_y[1] if roi_y else h_full
        xs = roi_x[0] if roi_x else 0
        xe = roi_x[1] if roi_x else w_full
        frames = det[:, ys:ye, xs:xe].astype(np.float32)

    if specific_frame_idx is None:
        idx = n_frames // 2
    else:
        idx = int(specific_frame_idx)
        if idx < 0 or idx >= n_frames:
            idx = n_frames // 2
    raw_specific = frames[idx].copy()

    if subtract_background:
        background = np.median(frames, axis=0, overwrite_input=True).astype(np.float32)
        frames -= background[np.newaxis, :, :]
        specific_2d = raw_specific - background
    else:
        specific_2d = raw_specific

    sum_2d = frames.sum(axis=0)
    if normalize_sum:
        sum_2d = sum_2d / max(1, n_frames)

    del frames, raw_specific
    return sum_2d, specific_2d, n_frames, idx
```

Thread it through `build_raw_volumes` — add a `subtract_background: bool = True` parameter (place it right after `normalize_sum`) and pass it into the `process_raw_scan(...)` call:

```python
        try:
            sum_2d, spec_2d, _nf, spec_idx = process_raw_scan(
                h5p, detector_path, roi_x, roi_y, specific_frame_idx, normalize_sum,
                subtract_background,
            )
        except (KeyError, OSError, ValueError):
            continue
```

Add the `STAGE` param (in the "Alignment" group, after `normalize_sum`):

```python
        Param(
            "subtract_background",
            ParamType.BOOL,
            "Subtract background",
            default=True,
            advanced=True,
            group="Alignment",
            help=(
                "Subtract a per-pixel median background (across the scan's frames) before "
                "summing. On for the standard rocking sum; turn off for a plain intensity sum "
                "(e.g. a mosa-scan topograph that keeps the background)."
            ),
        ),
```

In `run()`, pass the param into `build_raw_volumes` (add the argument after `bool(p["normalize_sum"])`):

```python
    sum_vol, spec_vol, samy_used, samz_used, names_used, spec_idx = build_raw_volumes(
        keep_paths,
        keep_samy,
        keep_samz,
        p["detector_path"],
        roi_x,
        roi_y,
        spec_cfg,
        bool(p["normalize_sum"]),
        bool(p["subtract_background"]),
        progress=lambda fr, m: progress(0.1 + 0.5 * fr, m),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_stage_rocking.py -q`
Expected: PASS (new test + the existing `process_raw_scan` / run tests, which default `subtract_background=True` and are unchanged).

- [ ] **Step 5: Update docs**

In `docs/Usage.md` (rocking section) document the "Subtract background" toggle. In `docs/Codebase.md` update the `process_raw_scan` / `build_raw_volumes` signatures.

- [ ] **Step 6: Commit**

```bash
git add dfxm/stages/rocking.py tests/test_stage_rocking.py docs/Usage.md docs/Codebase.md
git commit -m "feat(rocking): optional per-pixel background subtraction (subtract_background)

<trailers>"
```

---

## Task 3: Rocking — mosa-scan source (`source_scan`)

**Files:**
- Modify: `dfxm/stages/rocking.py` (`STAGE` param, `run()` source branch + output naming + titles, `figures()` titles)
- Test: `tests/test_stage_rocking.py`

**Interfaces:**
- Consumes: `process_raw_scan(..., subtract_background)` from Task 2.
- Produces: with `source_scan="mosaicity"`, `run()` builds the raw volumes from the `mosa_pattern` folders (every matched folder is a layer, no samz-union masking), aligns to the mosa reference, and — when the output name/dir are still the rocking defaults — writes `aligned_raw_mosa_volumes.h5` under an `aligned_raw_mosa_volumes/` folder. Product titles read "Mosa-integrated …".

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_rocking.py`:

```python
def test_run_mosaicity_source_builds_mosa_volume(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # three mosa layers, each with its own frame stack; these ARE the layers
    for i, z in enumerate([0.0, 0.001, 0.002]):
        _write_motor_folder(str(raw), f"mosa__{i + 1}", 0.0001 * i, z, frames=_rng_frames(i))
    res = RK.run(
        {
            "raw_root": str(raw),
            "source_scan": "mosaicity",
            "mosa_pattern": "mosa__*",
            "pixel_size_x_um": 0.152,
            "pixel_size_y_um": 0.385,
            "save_layers": False,
            "save_animation": False,
            "save_topview": False,
        }
    )
    assert res.n_layers_used == 3
    # default output auto-renamed so it never clobbers the rocking file
    assert res.aligned_path.endswith("aligned_raw_mosa_volumes.h5")
    assert os.path.exists(res.aligned_path)
    assert res.volume_shape[0] == 3
    # source-aware product title
    assert any(d.name == "raw_sum_intensity" for d in res.datasets)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_rocking.py::test_run_mosaicity_source_builds_mosa_volume -v`
Expected: FAIL — `source_scan` is not a param, so the mosa-only folder set is treated as rocking input and the run raises `StageUserError` ("no rocking folders matching '*'") or writes the rocking-named file.

- [ ] **Step 3: Write minimal implementation**

Add the `STAGE` param (in the "Data layout" group, right after `rocking_pattern`):

```python
        Param(
            "source_scan",
            ParamType.ENUM,
            "Source scan",
            default="rocking",
            choices=("rocking", "mosaicity"),
            advanced=True,
            group="Data layout",
            help=(
                "Which scans' detector frames are summed into the raw volume. 'rocking' uses the "
                "rocking scans (within the mosa/strain Z range); 'mosaicity' sums each mosa scan's "
                "frames — one mosa folder per layer — as a DFXM topograph. Run once per source to "
                "build both."
            ),
        ),
```

Add two source-aware title helpers near the top of `rocking.py` (after `_noop`):

```python
def _sum_title(source: str) -> str:
    return "Mosa-integrated Sum Intensity" if source == "mosaicity" else (
        "Background-subtracted Sum Intensity"
    )


def _spec_title(source: str, idx: int) -> str:
    base = "Mosa-integrated Frame" if source == "mosaicity" else "Background-subtracted Frame"
    return f"{base} {idx}"
```

In `run()`, after resolving `p`, read the source and branch the output-dir default. Replace:

```python
    out_dir = p["output_dir"] or os.path.join(raw_root, "aligned_raw_rocking_volumes")
```

with:

```python
    source = p.get("source_scan", "rocking")
    default_dir = "aligned_raw_mosa_volumes" if source == "mosaicity" else "aligned_raw_rocking_volumes"
    out_dir = p["output_dir"] or os.path.join(raw_root, default_dir)
```

Change the mosa-reference read to keep the folder names (needed as the layer list). Replace:

```python
    mosa_samy, mosa_samz, _ = _motors(raw_root, p["mosa_pattern"], p["samy_path"], p["samz_path"])
```

with:

```python
    mosa_samy, mosa_samz, mosa_names = _motors(
        raw_root, p["mosa_pattern"], p["samy_path"], p["samz_path"]
    )
```

Replace the whole "2. samz union range" + "3. rocking scans within the samz union range" block (from the `# 2. samz union range` comment down to and including the `keep_samy, keep_samz = keep_samy[order], keep_samz[order]` line) with a source branch:

```python
    # 2/3. choose the layers to process
    if source == "mosaicity":
        # the mosa scans themselves are the layers (no samz-union masking)
        keep_paths = [os.path.join(raw_root, n) for n in mosa_names]
        keep_samy, keep_samz = np.asarray(mosa_samy), np.asarray(mosa_samz)
    else:
        # samz union range (mosa ∪ strain)
        _, strain_samz, _ = _motors(raw_root, p["strain_pattern"], p["samy_path"], p["samz_path"])
        all_samz = np.concatenate([mosa_samz, strain_samz]) if len(strain_samz) else mosa_samz
        z_min, z_max = float(all_samz.min()), float(all_samz.max())

        progress(0.06, "reading rocking motor positions")
        rock_samy, rock_samz, rock_names = _motors(
            raw_root, p["rocking_pattern"], p["samy_path"], p["samz_path"]
        )
        if len(rock_names) == 0:
            raise StageUserError(
                f"no rocking folders matching {p['rocking_pattern']!r} in {raw_root}",
                hint="Check 'Rocking pattern' against the scan folder names under the raw root.",
            )
        rock_paths = [os.path.join(raw_root, n) for n in rock_names]
        mask = (rock_samz >= z_min - tol) & (rock_samz <= z_max + tol)
        keep_paths = [pp for pp, m in zip(rock_paths, mask) if m]
        keep_samy, keep_samz = rock_samy[mask], rock_samz[mask]
        if not keep_paths:
            raise StageUserError(
                f"no rocking scans fall in samz union [{z_min:.6f}, {z_max:.6f}] mm (tol={tol})",
                hint=(
                    "Loosen 'samz tolerance' or check that the rocking scans "
                    "cover the mosaicity/strain Z range."
                ),
            )

    order = np.argsort(keep_samz)
    keep_paths = [keep_paths[i] for i in order]
    keep_samy, keep_samz = keep_samy[order], keep_samz[order]
```

Branch the aligned filename at save time. Replace:

```python
    if p["save_aligned_h5"]:
        aligned_path = os.path.join(out_dir, p["aligned_h5_name"])
```

with:

```python
    if p["save_aligned_h5"]:
        aligned_name = p["aligned_h5_name"]
        if source == "mosaicity" and aligned_name == STAGE.defaults()["aligned_h5_name"]:
            aligned_name = "aligned_raw_mosa_volumes.h5"
        aligned_path = os.path.join(out_dir, aligned_name)
```

Make the two `_render(...)` calls use source-aware titles. Replace the sum title `"Background-subtracted Sum Intensity"` with `_sum_title(source)` and the specific title `f"Background-subtracted Frame {spec_idx}"` with `_spec_title(source, spec_idx)`.

Finally make `figures()` mirror the source-aware titles. In `figures()` read the source and use the helpers:

```python
    source = str(params.get("source_scan", "rocking"))
```

and replace the two title assignments:

```python
        if ds_key == "sum_intensity":
            cbar_label = f"Sum intensity {sum_tag}"
            title = _sum_title(source)
        else:
            cbar_label = _PRODUCT_CBAR.get(ds_key, prod.name)
            if ds_key == "specific_frame" and result.specific_frame_idx is not None:
                title = _spec_title(source, result.specific_frame_idx)
            else:
                title = _PRODUCT_TITLE.get(ds_key, prod.name)
```

(Delete the now-superseded `_PRODUCT_TITLE`-only title lines they replace.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_stage_rocking.py -q`
Expected: PASS (new mosaicity-source test + all existing rocking tests, whose default `source_scan="rocking"` path is unchanged).

- [ ] **Step 5: Update docs**

In `docs/Usage.md` (rocking section) document the "Source scan" selector, the mosaicity workflow (run once per source), and the auto-named `aligned_raw_mosa_volumes.h5`. In `docs/Codebase.md` update the `rocking.run()` description + note `_sum_title`/`_spec_title`.

- [ ] **Step 6: Commit**

```bash
git add dfxm/stages/rocking.py tests/test_stage_rocking.py docs/Usage.md docs/Codebase.md
git commit -m "feat(rocking): mosa-scan source (source_scan) building a parallel raw volume

<trailers>"
```

---

## Task 4: Slices + plotting — consume the mosa-sum volume

**Files:**
- Modify: `dfxm/stages/slices.py` (`STAGE` params, `_STD_VOLUMES`, `_standard_volumes`, `prepare_volume` titles)
- Modify: `dfxm/common/plotting.py` (`GROUP_BY_KIND`)
- Modify: `gui/bindings.py` (`_ALIGNED_MOSA` constant + slices override)
- Test: `tests/test_stage_slices.py`

**Interfaces:**
- Consumes: an aligned raw volume with `sum_intensity` / `specific_frame` datasets (identical schema to rocking output, whatever the source).
- Produces: two new sliceable fields `raw_mosa_sum` / `raw_mosa_specific` loaded from `aligned_mosa_file` via `include_mosa_sum` / `include_mosa_specific`; both map to the `"raw"` colormap group.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_slices.py` (a synthetic mosa file uses the same schema the slices `_setup` already writes for the rocking file):

```python
def test_run_includes_mosa_raw_field(tmp_path):
    proc, raw = _setup(tmp_path)
    rng = np.random.default_rng(1)
    with h5py.File(proc / "aligned_raw_mosa_volumes.h5", "w") as f:
        f.create_dataset("sum_intensity", data=rng.standard_normal((L, NY, NX)).astype(np.float32))
        f.create_dataset("specific_frame", data=rng.standard_normal((L, NY, NX)).astype(np.float32))
        f.create_dataset("z_uniform_um", data=np.arange(L, dtype=np.float32))
        f.attrs["scale_x_um_per_px"] = 0.152
        f.attrs["scale_y_um_per_px"] = 0.385
        f.attrs["scale_z_um_per_px"] = 1.0
        f.attrs["specific_frame_idx"] = 2
    slices_json = (
        '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
        '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
    )
    res = S.run(
        {
            "aligned_mosa_file": str(proc / "aligned_raw_mosa_volumes.h5"),
            "include_mosa_sum": True,
            "include_mosa_specific": False,
            # keep the run small: turn the standard volumes off
            "include_mosa_com_chi": False,
            "include_mosa_fwhm_chi": False,
            "include_mosa_com_mu": False,
            "include_mosa_fwhm_mu": False,
            "include_strain": False,
            "include_raw_sum": False,
            "include_raw_specific": False,
            "slices_json": slices_json,
            "output_dir": str(tmp_path / "sl"),
        }
    )
    assert "raw_mosa_sum" in res.volume_ids
    with h5py.File(res.output_h5, "r") as f:
        assert f["raw_mosa_sum"].attrs["kind"] == "raw_mosa_sum"
        assert f["raw_mosa_sum"].attrs["title"] == "Mosa-integrated Sum Intensity"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_slices.py::test_run_includes_mosa_raw_field -v`
Expected: FAIL with `KeyError: 'aligned_mosa_file'` (param missing from `_standard_volumes`' `file_keys` / defaults).

- [ ] **Step 3: Write minimal implementation**

In `dfxm/common/plotting.py`, extend `GROUP_BY_KIND`:

```python
GROUP_BY_KIND: dict[str, str] = {
    "mosa_com": "mosa_com",
    "mosa_fwhm": "mosa_fwhm",
    "strain": "strain",
    "raw_sum": "raw",
    "raw_specific": "raw",
    "raw_mosa_sum": "raw",
    "raw_mosa_specific": "raw",
}
```

In `dfxm/stages/slices.py`, append two rows to `_STD_VOLUMES`:

```python
_STD_VOLUMES = (
    ("include_mosa_com_chi", "stacked", "mosa_volume_file", "chi/Center of mass", "mosa_com"),
    ("include_mosa_fwhm_chi", "stacked", "mosa_volume_file", "chi/FWHM", "mosa_fwhm"),
    ("include_mosa_com_mu", "stacked", "mosa_volume_file", "mu/Center of mass", "mosa_com"),
    ("include_mosa_fwhm_mu", "stacked", "mosa_volume_file", "mu/FWHM", "mosa_fwhm"),
    ("include_strain", "stacked", "strain_volume_file", "strain", "strain"),
    ("include_raw_sum", "aligned", "aligned_rocking_file", "sum_intensity", "raw_sum"),
    ("include_raw_specific", "aligned", "aligned_rocking_file", "specific_frame", "raw_specific"),
    ("include_mosa_sum", "aligned", "aligned_mosa_file", "sum_intensity", "raw_mosa_sum"),
    ("include_mosa_specific", "aligned", "aligned_mosa_file", "specific_frame", "raw_mosa_specific"),
)
```

Add the `aligned_mosa_file` PATH param (immediately after `aligned_rocking_file` in `STAGE.params`):

```python
        Param(
            "aligned_mosa_file",
            ParamType.PATH,
            "Aligned mosa volume",
            must_exist=True,
            help=(
                "The aligned mosa-sum volume (aligned_raw_mosa_volumes.h5) from the rocking stage "
                "run with Source scan = mosaicity. Leave blank to skip the mosa raw fields."
            ),
        ),
```

Add the two toggles (in the "Quantities" group, after `include_raw_specific`):

```python
        Param(
            "include_mosa_sum",
            ParamType.BOOL,
            "Slice mosa sum",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the mosa-scan summed intensity volume (from the aligned mosa file).",
        ),
        Param(
            "include_mosa_specific",
            ParamType.BOOL,
            "Slice mosa specific",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the mosa-scan specific-frame intensity volume.",
        ),
```

Add the file key in `_standard_volumes`:

```python
    file_keys = {
        "mosa_volume_file": (p["mosa_volume_file"], p["raw_root"], p["mosa_pattern"]),
        "strain_volume_file": (p["strain_volume_file"], p["raw_root"], p["strain_pattern"]),
        "aligned_rocking_file": (p["aligned_rocking_file"], "", ""),
        "aligned_mosa_file": (p["aligned_mosa_file"], "", ""),
    }
```

Add the two titles in `prepare_volume`'s `titles` dict:

```python
        "raw_mosa_sum": ("Mosa-integrated Sum Intensity", "Sum intensity (a.u.)", ""),
        "raw_mosa_specific": (
            f"Mosa-integrated Frame {int(extra.get('specific_frame_idx', -1))}",
            "Intensity (a.u.)",
            f"_frame{int(extra.get('specific_frame_idx', -1))}",
        ),
```

In `gui/bindings.py`, add the constant next to `_ALIGNED_ROCKING` and pre-fill the slices override:

```python
_ALIGNED_MOSA = "aligned_raw_mosa_volumes.h5"
```

and in the `if stage_name == "slices":` branch add:

```python
            aligned_mosa_file=os.path.join(proc, _ALIGNED_MOSA) if proc else "",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_stage_slices.py tests/test_param_metadata.py -q`
Expected: PASS (new field test + the metadata guardrail, which checks the new params have `help`/`group`).

- [ ] **Step 5: Update docs**

In `docs/Usage.md` (slices Quantities + inputs) document the "Aligned mosa volume" input and the two mosa quantity toggles. In `docs/Codebase.md` add the new params, the `_STD_VOLUMES` rows, the `raw_mosa_*` kinds, and the `GROUP_BY_KIND` additions.

- [ ] **Step 6: Commit**

```bash
git add dfxm/stages/slices.py dfxm/common/plotting.py gui/bindings.py tests/test_stage_slices.py docs/Usage.md docs/Codebase.md
git commit -m "feat(slices): consume the mosa-sum raw volume as raw_mosa_sum/raw_mosa_specific fields

<trailers>"
```

---

## Task 5: Slices — replot core (`replot_catalog`, `render_replot`, shared rebuild helper)

**Files:**
- Modify: `dfxm/stages/slices.py` (add `ReplotEntry`, `_rebuild_plane_figure`, `replot_catalog`, `render_replot`; refactor `figures()` to use the helper)
- Test: `tests/test_stage_slices.py`

**Interfaces:**
- Consumes: an `oblique_slices.h5` written by `run()` (volume groups → slice subgroups → `slices`/`u_um`/`v_um`/`offsets_um`, with per-volume attrs `kind`/`cmap`/`title`/`cbar_label`/`vmin`/`vmax`).
- Produces:
  - `replot_catalog(h5_path: str) -> list[ReplotEntry]` where `ReplotEntry(volume_id: str, slice_name: str, n_planes: int, offsets_um: list[float])`.
  - `render_replot(h5_path, selections, style, clim, out_dir, *, dpi=150) -> list[str]` with `selections: list[tuple[str, str, list[int] | None]]` (vid, slice, plane_idxs — `None` = all planes) and `clim: tuple[float | None, float | None] | None`. Writes `{out_dir}/{slice}/{vid}[__pNNN_{off}um].png`; returns written paths.
  - `_rebuild_plane_figure(h5_path, vid, sname, k, style, *, clim=None) -> Figure`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_slices.py` (a small consolidated file mirroring the profiles test fixture, so the core is exercised without a full run):

```python
def _write_mini_consolidated(path):
    """Two fields sharing one slice with 3 planes; raw-group + strain-group attrs."""
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    offsets = np.array([-1.0, 0.0, 1.0])
    with h5py.File(path, "w") as f:
        for vid, kind, cmap in (("raw_sum", "raw_sum", "gray"), ("strain", "strain", "RdBu_r")):
            g = f.create_group(vid)
            g.attrs["kind"] = kind
            g.attrs["cmap"] = cmap
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "value"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("plane_a")
            sg.create_dataset(
                "slices", data=np.zeros((3, v.size, u.size), dtype=np.float32)
            )
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offsets)


def test_replot_catalog_enumerates_volumes_slices_planes(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    cat = S.replot_catalog(str(h5))
    by_vid = {(e.volume_id, e.slice_name): e for e in cat}
    assert set(by_vid) == {("raw_sum", "plane_a"), ("strain", "plane_a")}
    assert by_vid[("strain", "plane_a")].n_planes == 3
    assert by_vid[("strain", "plane_a")].offsets_um == [-1.0, 0.0, 1.0]


def test_render_replot_writes_selected_planes_under_subfolders(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    out = tmp_path / "replots"
    # strain: only planes 0 and 2; raw_sum: all planes (None)
    written = S.render_replot(
        str(h5),
        [("strain", "plane_a", [0, 2]), ("raw_sum", "plane_a", None)],
        style=None,
        clim=None,
        out_dir=str(out),
    )
    assert len(written) == 2 + 3
    assert all(os.path.exists(p) for p in written)
    # per-slice subfolder layout
    assert all(os.path.basename(os.path.dirname(p)) == "plane_a" for p in written)


def test_render_replot_clim_override_changes_norm(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    fig = S._rebuild_plane_figure(str(h5), "strain", "plane_a", 1, style=None, clim=(-5.0, 5.0))
    im = fig.axes[0].images[0]
    assert im.norm.vmin == -5.0 and im.norm.vmax == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_slices.py::test_replot_catalog_enumerates_volumes_slices_planes -v`
Expected: FAIL with `AttributeError: module 'dfxm.stages.slices' has no attribute 'replot_catalog'`.

- [ ] **Step 3: Write minimal implementation**

In `dfxm/stages/slices.py`, add a `ReplotEntry` dataclass (next to `SlicesResult`):

```python
@dataclass
class ReplotEntry:
    volume_id: str
    slice_name: str
    n_planes: int
    offsets_um: list[float] = field(default_factory=list)
```

Add the shared rebuild helper + the two public functions (after `figures()` is fine; place them before `_main`):

```python
def _rebuild_plane_figure(h5_path, vid, sname, k, style, *, clim=None) -> Figure:
    """Rebuild one plane's slice figure from an oblique_slices.h5 group.

    Shared by :func:`figures` (catalog/export) and :func:`render_replot` so the
    prep-from-attrs reconstruction lives in exactly one place. ``clim`` is an
    optional ``(vmin, vmax)`` override; ``None`` entries keep the stored value.
    """
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
    if clim is not None:
        vmin_o, vmax_o = clim
        if vmin_o is not None:
            prep["vmin"] = float(vmin_o)
        if vmax_o is not None:
            prep["vmax"] = float(vmax_o)
    prep["cmap_name"] = resolve_cmap(style, GROUP_BY_KIND.get(kind), fallback=prep["cmap_name"])
    prep["group"] = GROUP_BY_KIND.get(kind)
    return build_slice_figure(prep, {"name": sname}, s2d, u, v, offset_um=off, style=style)


def replot_catalog(h5_path: str) -> list[ReplotEntry]:
    """List every (volume_id, slice_name, n_planes, offsets_um) in an oblique_slices.h5."""
    entries: list[ReplotEntry] = []
    with h5py.File(h5_path, "r") as f:
        for vid in f.keys():
            vg = f[vid]
            if not isinstance(vg, h5py.Group):
                continue
            for sname in vg.keys():
                sg = vg[sname]
                if not (isinstance(sg, h5py.Group) and "slices" in sg):
                    continue
                offsets = [float(o) for o in sg["offsets_um"][:]]
                entries.append(
                    ReplotEntry(vid, sname, int(sg["slices"].shape[0]), offsets)
                )
    return entries


def render_replot(
    h5_path: str,
    selections: list[tuple[str, str, list[int] | None]],
    style: PlotStyle | None,
    clim: tuple[float | None, float | None] | None,
    out_dir: str,
    *,
    dpi: int = 150,
) -> list[str]:
    """Rebuild + save the selected planes (appearance-only; no resampling).

    ``selections`` is a list of ``(volume_id, slice_name, plane_idxs)`` where
    ``plane_idxs`` is ``None`` for all planes. PNGs are written under
    ``{out_dir}/{slice_name}/`` mirroring the slices run layout; returns the
    written paths.
    """
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
            fig = _rebuild_plane_figure(h5_path, vid, sname, k, style, clim=clim)
            if entry.n_planes == 1:
                png = os.path.join(slice_dir, f"{vid}.png")
            else:
                png = os.path.join(
                    slice_dir, f"{vid}__p{k:03d}_{entry.offsets_um[k]:+08.2f}um.png"
                )
            fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight")
            written.append(png)
    return written
```

Refactor `figures()` to route its build closure through the shared helper. Replace the inner `def build(style, ...): …` body with:

```python
                    def build(style, vid=vid, sname=sname, k=k):
                        return _rebuild_plane_figure(result.output_h5, vid, sname, k, style)
```

(The surrounding `prep = {...}` / `kind = ...` lines that only fed the old closure can be removed; `_rebuild_plane_figure` reads them from the file itself. Keep the `FigureSpec(...)` append unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_stage_slices.py -q`
Expected: PASS (new replot-core tests + the existing `figures()`-based tests, which now go through `_rebuild_plane_figure`).

- [ ] **Step 5: Update docs**

In `docs/Codebase.md` (dfxm/stages/slices) document `ReplotEntry`, `replot_catalog`, `render_replot`, and `_rebuild_plane_figure` (shared by `figures()` + replot). No `Usage.md` change here — the user-facing surface is the dialog (Task 6).

- [ ] **Step 6: Commit**

```bash
git add dfxm/stages/slices.py tests/test_stage_slices.py docs/Codebase.md
git commit -m "feat(slices): replot core — replot_catalog + render_replot from oblique_slices.h5

<trailers>"
```

---

## Task 6: GUI — `SliceReplotDialog` + "Replot…" button

**Files:**
- Create: `gui/widgets/slice_replot.py`
- Modify: `gui/stage_view.py` (button in the slices branch + `_on_replot` slot)
- Test: `tests/test_gui_slice_replot.py` (new)

**Interfaces:**
- Consumes: `slices.replot_catalog`, `slices.render_replot` (Task 5); the session `PlotStyle` from `window.global_plot_style()`.
- Produces: `SliceReplotDialog(h5_path: str, style, out_default: str, parent=None)` with a `_tree` (`QTreeWidget`) populated from the catalog and a `render()` that calls `slices.render_replot`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_slice_replot.py`:

```python
"""Offscreen construction test for the slices Replot dialog (delegates rendering
to the tested Qt-free core in dfxm.stages.slices)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


def _mini(path):
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    with h5py.File(path, "w") as f:
        for vid in ("raw_sum", "strain"):
            g = f.create_group(vid)
            g.attrs["kind"] = vid
            g.attrs["cmap"] = "gray"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "v"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("plane_a")
            sg.create_dataset("slices", data=np.zeros((2, v.size, u.size), dtype=np.float32))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0, 1.0]))


def test_dialog_populates_tree_and_renders(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    out = tmp_path / "replots"
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(out))
    # two volume groups at the top level
    assert dlg._tree.topLevelItemCount() == 2
    # select everything and render straight through the core
    written = dlg.render_selection(str(out))
    assert written and all(os.path.exists(p) for p in written)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_gui_slice_replot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gui.widgets.slice_replot'`.

- [ ] **Step 3: Write minimal implementation**

Create `gui/widgets/slice_replot.py`:

```python
"""Slice replot dialog (built lazily on demand).

Reads an oblique_slices.h5 straight from disk and re-renders selected planes
with the current publication style + an optional colour-limit override — no
resampling, and no prior stage run required (works from a cold start). All the
figure work happens in the Qt-free core (dfxm.stages.slices); this dialog is a
thin shell around slices.replot_catalog / slices.render_replot.
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

from dfxm.stages import slices as _sl


class SliceReplotDialog(QDialog):
    """Pick volumes/slices/planes from an oblique_slices.h5 and re-render PNGs."""

    def __init__(self, h5_path, style=None, out_default="", parent=None) -> None:
        super().__init__(parent)
        self._h5_path = h5_path
        self._style = style
        self.written: list[str] = []

        self.setWindowTitle(f"Replot slices — {os.path.basename(h5_path)}")

        # file field (browsable; defaults to the passed h5, editable for a cold start)
        self._file_edit = QLineEdit(h5_path)
        file_browse = QPushButton("Browse…")
        file_browse.clicked.connect(self._on_browse_h5)
        file_reload = QPushButton("Load")
        file_reload.clicked.connect(self._reload)
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Slices file:"))
        file_row.addWidget(self._file_edit, 1)
        file_row.addWidget(file_browse)
        file_row.addWidget(file_reload)

        # volume → slice → plane tree (checkable)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Volume / slice / plane"])
        self._tree.setColumnCount(1)

        # clim override
        self._vmin = QLineEdit()
        self._vmin.setPlaceholderText("vmin (blank = stored)")
        self._vmax = QLineEdit()
        self._vmax.setPlaceholderText("vmax (blank = stored)")
        clim_row = QHBoxLayout()
        clim_row.addWidget(QLabel("Colour limits:"))
        clim_row.addWidget(self._vmin)
        clim_row.addWidget(self._vmax)

        # output dir
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
        layout.addLayout(clim_row)
        layout.addLayout(out_row)
        layout.addLayout(btn_row)

        self._reload()

    # -- population -----------------------------------------------------------
    def _reload(self) -> None:
        self._h5_path = self._file_edit.text().strip()
        self._tree.clear()
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._status.setText("no such file")
            return
        try:
            catalog = _sl.replot_catalog(self._h5_path)
        except (OSError, KeyError) as exc:  # unreadable / not a slices file
            self._status.setText(f"cannot read: {exc}")
            return
        by_vid: dict[str, QTreeWidgetItem] = {}
        for entry in catalog:
            vtop = by_vid.get(entry.volume_id)
            if vtop is None:
                vtop = QTreeWidgetItem(self._tree, [entry.volume_id])
                vtop.setFlags(vtop.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                vtop.setCheckState(0, Qt.CheckState.Unchecked)
                by_vid[entry.volume_id] = vtop
            snode = QTreeWidgetItem(vtop, [entry.slice_name])
            snode.setFlags(snode.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            snode.setCheckState(0, Qt.CheckState.Unchecked)
            snode.setData(0, Qt.ItemDataRole.UserRole, (entry.volume_id, entry.slice_name))
            for k, off in enumerate(entry.offsets_um):
                leaf = QTreeWidgetItem(snode, [f"plane {k}  ({off:+.2f} µm)"])
                leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setCheckState(0, Qt.CheckState.Unchecked)
                leaf.setData(0, Qt.ItemDataRole.UserRole, (entry.volume_id, entry.slice_name, k))
        self._tree.expandAll()
        self._status.setText(f"{len(catalog)} slice group(s)")

    # -- selection → core -----------------------------------------------------
    def _selections(self):
        """Collect checked (vid, slice, plane_idxs|None) tuples from the tree."""
        sels = []
        for i in range(self._tree.topLevelItemCount()):
            vtop = self._tree.topLevelItem(i)
            for j in range(vtop.childCount()):
                snode = vtop.child(j)
                vid, sname = snode.data(0, Qt.ItemDataRole.UserRole)
                checked = [
                    snode.child(k).data(0, Qt.ItemDataRole.UserRole)[2]
                    for k in range(snode.childCount())
                    if snode.child(k).checkState(0) == Qt.CheckState.Checked
                ]
                if snode.checkState(0) == Qt.CheckState.Checked and not checked:
                    sels.append((vid, sname, None))  # whole slice = all planes
                elif checked:
                    sels.append((vid, sname, checked))
        return sels

    def _clim(self):
        def _f(edit):
            t = edit.text().strip()
            return float(t) if t else None

        try:
            vmin, vmax = _f(self._vmin), _f(self._vmax)
        except ValueError:
            return None
        if vmin is None and vmax is None:
            return None
        return (vmin, vmax)

    def render_selection(self, out_dir):
        """Render currently-checked planes into *out_dir*; returns written paths."""
        self.written = _sl.render_replot(
            self._h5_path, self._selections(), self._style, self._clim(), out_dir
        )
        return self.written

    # -- slots ----------------------------------------------------------------
    def _on_render(self) -> None:
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            self._status.setText("set an output dir")
            return
        sels = self._selections()
        if not sels:
            self._status.setText("nothing selected")
            return
        written = self.render_selection(out_dir)
        self._status.setText(f"wrote {len(written)} PNG(s) → {out_dir}")

    def _on_browse_h5(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open oblique_slices.h5", "", "HDF5 (*.h5)")
        if path:
            self._file_edit.setText(path)
            self._reload()

    def _on_browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output directory")
        if path:
            self._out_edit.setText(path)
```

Wire it into `gui/stage_view.py`. In `__init__`, alongside the profiles `Pick line…` guard, add a slices `Replot…` button:

```python
        self._replot_btn: QPushButton | None = None
        if stage_name == "slices":
            self._replot_btn = QPushButton("Replot…")
            self._replot_btn.clicked.connect(self._on_replot)
            btn_row.addWidget(self._replot_btn)
```

Add the slot (near `_on_pick_line`):

```python
    # -- slices interactive replot (lazy) ---------------------------------
    def _on_replot(self) -> None:
        import time
        from dataclasses import replace

        vals = self._form.values()
        out_dir = vals.get("output_dir", "") or os.path.join(
            os.path.dirname(
                vals.get("mosa_volume_file", "") or vals.get("strain_volume_file", "") or "."
            ),
            "oblique_slices",
        )
        h5 = os.path.join(out_dir, vals.get("output_h5_name", "") or "oblique_slices.h5")

        window = self.window()
        style = window.global_plot_style() if hasattr(window, "global_plot_style") else None

        replots_dir = os.path.join(out_dir, "replots", time.strftime("%Y%m%d-%H%M%S"))

        from .widgets.slice_replot import SliceReplotDialog  # imported on demand

        dlg = SliceReplotDialog(
            h5, style=replace(style) if style is not None else None,
            out_default=replots_dir, parent=self,
        )
        dlg.exec()
        if dlg.written:
            self._log.append(f"Replotted {len(dlg.written)} PNG(s) → {replots_dir}")
            self._tabs.setCurrentWidget(self._log)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_gui_slice_replot.py -q`
Expected: PASS.

- [ ] **Step 5: Update docs + gui_smoke**

In `docs/Usage.md` add a "Replotting slices without re-running" subsection under the slices stage: the Replot… button reads an `oblique_slices.h5` from disk (works after a restart), lets you tick volumes/slices/planes, override vmin/vmax, and writes into a timestamped `replots/<stamp>/<slice>/` folder. In `docs/Codebase.md` (gui/widgets) add `slice_replot.SliceReplotDialog` and the `stage_view._on_replot` wiring. Add a numbered step to `tests/gui_smoke.py` that opens the slices stage, clicks "Replot…", and renders one plane.

- [ ] **Step 6: Commit**

```bash
git add gui/widgets/slice_replot.py gui/stage_view.py tests/test_gui_slice_replot.py tests/gui_smoke.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): slices Replot… subwidget — pick + re-render planes from oblique_slices.h5

<trailers>"
```

---

## Task 7: Profiles — per-job field selection (`fields` / `reference`)

**Files:**
- Modify: `dfxm/stages/profiles.py` (`_collect`; the default `jobs_json` help text)
- Test: `tests/test_stage_profiles.py`

**Interfaces:**
- Consumes: an `oblique_slices.h5` with ≥2 fields sharing a slice.
- Produces: each job may carry optional `"fields": [vid, …]` (restricts + orders the profiled fields for that job) and `"reference": vid` (top image), each falling back to the global `volume_ids` / `reference_volume_id` when absent. Read inside `_collect`, so both `run()` and `figures()` honour it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_profiles.py` (reuses the module's `_write_consolidated`, which writes `raw_sum` + `strain`):

```python
def test_run_per_job_fields_restricts_profiled_fields(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1,"fig_name":"only_strain","fields":["strain"]}]'
    )
    res = PR.run(
        {"consolidated_h5": str(h5), "mode": "parameter", "jobs_json": jobs, "output_dir": str(out)}
    )
    assert res.jobs[0].fields == ["strain"]  # raw_sum excluded for this job
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_profiles.py::test_run_per_job_fields_restricts_profiled_fields -v`
Expected: FAIL — `_collect` ignores the job's `fields`, so both `raw_sum` and `strain` are profiled and `res.jobs[0].fields == ["strain", "raw_sum"]` (order per the global rule).

- [ ] **Step 3: Write minimal implementation**

In `dfxm/stages/profiles.py`, read the per-job overrides at the top of `_collect` (right after `present` is computed). Replace:

```python
def _collect(f, job, p, ref_pref, restrict):
    name = job["name"]
    present = volume_ids_with_slice(f, name)
    if not present:
        raise KeyError(f"slice {name!r} not present in any field group")
    ref_id = _pick_reference_id(present, ref_pref)
```

with:

```python
def _collect(f, job, p, ref_pref, restrict):
    name = job["name"]
    present = volume_ids_with_slice(f, name)
    if not present:
        raise KeyError(f"slice {name!r} not present in any field group")
    # per-job overrides fall back to the global reference / restrict
    job_ref = job.get("reference") or ref_pref
    job_fields = job.get("fields") or restrict
    ref_id = _pick_reference_id(present, job_ref)
```

and change the field-iteration line from `restrict` to `job_fields`:

```python
    for vid in _ordered_field_ids(present, ref_id, job_fields):
```

Update the `jobs_json` param `help` to mention the new keys (append to the existing text):

```python
                "JSON list of profile jobs: slice name, plane offset, line start/end in µm "
                "('start_uv'/'end_uv'), and band width in pixels. Optional per-job 'fields' "
                "(list of field ids to profile, in order) and 'reference' (top image) override "
                "the global Fields/Reference. Easiest filled by 'Pick line…'."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_stage_profiles.py -q`
Expected: PASS (new per-job test + existing tests, whose jobs omit `fields`/`reference` and keep the global behaviour).

- [ ] **Step 5: Update docs**

In `docs/Usage.md` (profiles / Jobs JSON) document per-job `fields` and `reference`. In `docs/Codebase.md` update the `_collect` description to note the per-job override + fallback.

- [ ] **Step 6: Commit**

```bash
git add dfxm/stages/profiles.py tests/test_stage_profiles.py docs/Usage.md docs/Codebase.md
git commit -m "feat(profiles): per-job field/reference selection in jobs_json

<trailers>"
```

---

## Task 8: GUI — field checkboxes in the Pick-line dialog

**Files:**
- Modify: `gui/widgets/line_picker.py` (add a field checklist; include `fields` in `result`)
- Modify: `gui/viewers.py` (`inject_line_into_jobs` gains a `fields` arg — this is where it is defined, at line 60)
- Modify: `gui/stage_view.py` (`_on_pick_line` unpacks the 4-tuple and passes `fields`)
- Test: `tests/test_gui_line_picker_fields.py` (new)

**Interfaces:**
- Consumes: `profiles.volume_ids_with_slice` (already used by the picker); Task 7's per-job `fields`.
- Produces: `LinePickerDialog.result` becomes `(start_uv, end_uv, offset_um, fields)` where `fields` is the list of ticked field ids (all present, in catalog order, when none are unticked).

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_line_picker_fields.py`:

```python
"""Offscreen test: the Pick-line dialog exposes field checkboxes and returns them."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def _mini(path):
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    with h5py.File(path, "w") as f:
        for vid in ("raw_sum", "strain"):
            g = f.create_group(vid)
            g.attrs["kind"] = vid
            g.attrs["cmap"] = "gray"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "v"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("oblique_full")
            sg.create_dataset("slices", data=np.zeros((1, v.size, u.size), dtype=np.float32))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0]))


def test_picker_exposes_field_checkboxes(tmp_path):
    from gui.widgets.line_picker import LinePickerDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    dlg = LinePickerDialog(str(h5), "oblique_full")
    # one checkbox per present field, all checked by default
    assert set(dlg.selected_fields()) == {"raw_sum", "strain"}
    # unticking one narrows the returned set
    dlg._field_boxes["raw_sum"].setCheckState(Qt.CheckState.Unchecked)
    assert dlg.selected_fields() == ["strain"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_gui_line_picker_fields.py -v`
Expected: FAIL with `AttributeError: 'LinePickerDialog' object has no attribute 'selected_fields'`.

- [ ] **Step 3: Write minimal implementation**

In `gui/widgets/line_picker.py`, add `QCheckBox` to the imports, build one checkbox per present field, and expose `selected_fields()`. After `self._ref_id = pr._pick_reference_id(present, ref_pref)` store the present list:

```python
        self._present = present
```

In the widget layout (where the buttons/info are assembled), add a checklist (all checked by default), keyed by field id:

```python
        from PySide6.QtWidgets import QCheckBox

        self._field_boxes: dict[str, QCheckBox] = {}
        fields_row = QHBoxLayout()
        fields_row.addWidget(QLabel("Fields:"))
        for vid in self._present:
            box = QCheckBox(vid)
            box.setChecked(True)
            self._field_boxes[vid] = box
            fields_row.addWidget(box)
        fields_row.addStretch(1)
```

Add `fields_row` to the dialog's `QVBoxLayout` (`lay`) between the info label and the nav row — i.e. after `lay.addWidget(self._info)` and before `lay.addLayout(nav)`:

```python
        lay.addWidget(self._canvas, 1)
        lay.addWidget(self._info)
        lay.addLayout(fields_row)
        lay.addLayout(nav)
```

Add the accessor:

```python
    def selected_fields(self) -> list[str]:
        """Ticked field ids, in catalog order (all present when none unticked)."""
        return [vid for vid in self._present if self._field_boxes[vid].isChecked()]
```

In `_accept`, extend the exact existing result assignment
`self.result = (self._pts[0], self._pts[1], float(self._offsets[self._idx]))`
to a 4-tuple:

```python
        self.result = (
            self._pts[0], self._pts[1], float(self._offsets[self._idx]), self.selected_fields()
        )
```

Also update the module docstring line that documents the result shape (currently
"``(start_uv, end_uv, offset_um)``") to "``(start_uv, end_uv, offset_um, fields)``".

In `gui/stage_view.py` `_on_pick_line`, unpack the 4-tuple and pass `fields` when injecting the job. Replace the current `start, end, off = dlg.result` + call:

```python
        if dlg.exec() and dlg.result:
            start, end, off, fields = dlg.result
            new_jobs = inject_line_into_jobs(
                vals.get("jobs_json", "") or "[]", slice_name, start, end, off, fields=fields
            )
```

Extend `inject_line_into_jobs` in `gui/viewers.py` (defined at line 60) — add the `fields` param and write it onto the target job. Change the signature:

```python
def inject_line_into_jobs(
    jobs_json: str, slice_name: str, start_uv, end_uv, offset_um: float, fields=None
) -> str:
```

and add two lines just before the final `return json.dumps(jobs, indent=2)`:

```python
    target["end_uv"] = [round(float(end_uv[0]), 4), round(float(end_uv[1]), 4)]
    if fields is not None:
        target["fields"] = list(fields)
    return json.dumps(jobs, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_gui_line_picker_fields.py -q`
Expected: PASS.

- [ ] **Step 5: Update docs + gui_smoke**

In `docs/Usage.md` (profiles / Pick line…) note the field checklist and that ticking fewer fields writes a per-job `fields` list. In `docs/Codebase.md` update `line_picker.LinePickerDialog` (`selected_fields`, 4-tuple result) and `stage_view.inject_line_into_jobs` (`fields` arg). Update the profiles gui_smoke step to tick a field.

- [ ] **Step 6: Commit**

```bash
git add gui/widgets/line_picker.py gui/viewers.py gui/stage_view.py tests/test_gui_line_picker_fields.py tests/gui_smoke.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): field checkboxes in Pick line… → per-job profile fields

<trailers>"
```

---

## Final verification

- [ ] **Full suite:** `python3 -m pytest -q` — all green (the pre-existing 316-passed baseline plus the new tests).
- [ ] **Lint:** `ruff check .` — clean; `ruff format --check .` — clean.
- [ ] **GUI smoke:** run `tests/gui_smoke.py` manually and confirm the new Replot… and Pick-line-fields steps pass on screen.
- [ ] **Docs:** skim `docs/Usage.md` + `docs/Codebase.md` for the five features; confirm no stage/param/behaviour change shipped without its docs edit.

## Self-Review (against the spec)

**Spec coverage:** A→Task 1; B (core)→Task 5, B (dialog)→Task 6; C (subtract_background)→Task 2, C (source_scan)→Task 3; D→Task 4; E (core)→Task 7, E (picker)→Task 8. Testing/docs folded into each task; sequencing A→C→D→B→E preserved (Tasks 1,2,3,4,5,6,7,8 respect the cross-task dependencies: Task 3 uses Task 2's signature; Task 5 uses Task 1's layout; Task 6 uses Task 5; Task 8 uses Task 7).

**Type consistency:** `render_replot(h5_path, selections, style, clim, out_dir, *, dpi=150)` and `ReplotEntry(volume_id, slice_name, n_planes, offsets_um)` are used identically in Tasks 5 and 6; `_rebuild_plane_figure(..., *, clim=None)` shared by `figures()` + `render_replot`; `process_raw_scan(..., subtract_background=True)` from Task 2 is called with the positional `subtract_background` in Tasks 2 (via `build_raw_volumes`) and reused in Task 3; `LinePickerDialog.result` becomes a 4-tuple `(start, end, off, fields)` consumed by `stage_view._on_pick_line` + `inject_line_into_jobs(..., fields=None)` in Task 8; per-job `fields`/`reference` written by Task 8 are consumed by Task 7's `_collect`.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every test step shows the assertion.
