# Design — App-wide interactive ROI picker + per-quantity colour limits

- **Date:** 2026-07-09
- **Status:** approved (brainstorm), pending implementation plan
- **Scope:** single branch (one review cycle)
- **Fable planner review:** completed (agent `a0fc3a84d13a99e9e`) — endorsed Section A with a
  backward-compat refinement, recommended hand-rolled matplotlib over silx, enumerated 7 ROI
  entry points. User chose **everything in one branch**, **matplotlib** (no silx),
  **per-stage only** (no cross-stage crop propagation), **rocking run-time picker excluded**.

## Problem

Two user-reported issues with the DFXM GUI:

1. **ROI crop is confusing.** Region-of-interest crops are entered as four pixel-index boxes
   (`r0/r1` = rows/Y, `c0/c1` = cols/X). Maps render at true physical aspect — each Y pixel is
   ≈ 2.5× larger than X (`sx≈0.152`, `sy≈0.385 µm/px`) and Z-interpolation can inflate the Y
   pixel count — so a crop looks "huge in Y" and picking pixel bounds gives no intuition. The
   user wants an **interactive, darfix/silx-style ROI selector**: show a rendered plane, drag a
   rectangle, get the crop. This applies to **every** place an ROI is chosen, not just the
   replot dialogs.

2. **Colour limits are per colormap-group, should be per quantity.** In the slices replot
   dialog the per-kind clim rows key on the colormap *group* (`mosa_com`/`mosa_fwhm`/`strain`/
   `raw`), so χ-COM and μ-COM (both group `mosa_com`) share one vmin/vmax and cannot be scaled
   independently. Each distinct **quantity** (`volume_id`: `mosa_com_chi`, `mosa_com_mu`, …)
   should get its own clim row/scale.

## Decisions (settled during brainstorm)

- **Toolkit:** hand-rolled **matplotlib** `RectangleSelector` in an embedded `FigureCanvasQTAgg`
  — *not* silx. The app standardises on the explicit `matplotlib.figure.Figure` API embedded via
  `FigureCanvasQTAgg` (`gui/widgets/line_picker.py` is the precedent). silx would add a parallel
  plotting stack, a heavy dependency with lagging PySide6 support, and its only real advantage
  (darfix chrome) is the *drag-a-box interaction*, which `RectangleSelector` already delivers.
  Rendering the preview with the **same** mpl renderer as the exports makes WYSIWYG fidelity
  true by construction.
- **Picker aspect:** true physical aspect (WYSIWYG with the exports), no toggle.
- **Scope:** one branch covering the replot-dialog pickers, the χ/μ + raw_* clim split, **and**
  the run-time stage-param pickers (schema-driven), excluding rocking's run-time picker.
- **Cross-stage crops:** per-stage only. A picker fills only its own stage's boxes; no
  propagation to sibling co-registration stages.
- **Rocking run-time ROI:** excluded (draws on raw detector frames — a physically different grid
  from the darfix maps, potentially misleading). Rocking's *replot* picker (stored aligned data)
  is still included via the shared replot dialog.

## Section A — Per-quantity colour limits in slices replot

### Core: `dfxm/stages/slices.py::render_replot`

`ReplotEntry` (slices.py:448) already carries both `volume_id` and `group`, and `volume_id` is
`f"{kind}{suffix}"` with `_chi`/`_mu` suffixes (slices.py:623–629, 760). Change the clim key
from the colormap group to the per-quantity id, **with a two-key fallback** for backward
compatibility with any caller still passing a group-keyed dict (the documented contract since
the per-kind-clim project):

```python
# slices.py ~1175, was: clim_k = resolve_clim(clim, entry.group)
clim_k = resolve_clim(clim, entry.volume_id)
if clim_k is None:
    clim_k = resolve_clim(clim, entry.group)
```

- Colormap resolution stays on `GROUP_BY_KIND` (slices.py:1113–1114) — **unchanged**; χ and μ
  keep the *same* colormap.
- Legacy single-tuple clim still works via `resolve_clim`'s non-dict branch
  (`dfxm/common/figures.py:28–40`) — no change to `resolve_clim` itself.
- Update the `render_replot` docstring (slices.py:1160–1166), which currently enumerates the
  four group keys, to describe the volume_id keying + group fallback.

### GUI: `gui/widgets/slice_replot.py::_clim_groups`

Emit one clim row per distinct **`volume_id`** present in the catalog (currently one per
`entry.group`, slice_replot.py:222–233). Extend the label table so rows read, e.g.,
`Mosaicity COM (χ)`, `Mosaicity COM (μ)`, `Mosaicity FWHM (χ)`, `Mosaicity FWHM (μ)`, `Strain`,
`Raw sum intensity`, `Raw frame`, …; fall back to the bare `volume_id` for unknown ids. Keep the
existing stable ordering idea (known kinds first, then first-seen).

- `ClimGroupSection` (`gui/widgets/clim_section.py`) is **key-agnostic** and needs no change; its
  per-instance value cache simply stops matching old group keys (no persistence hazard — the
  cache is per-dialog-instance).

### raw_* split — keep it

The `raw_*` variants (`raw_sum`, `raw_specific`, `raw_mosa_sum`, `raw_mosa_specific`) currently
fold into one `raw` group. A summed volume vs a single frame differ in scale by ~the frame
count, so sharing one vmin/vmax was the same defect as χ/μ sharing. Splitting them is a fix, not
collateral damage. Realistic file → ~5–8 rows; acceptable.

### Not touched

The generic mosaicity/rocking/strain replot dialog is *already* per-component — it keys clim by
HDF5 dataset path (`mosaicity.py:341–349`, `rocking.py:985`). Run-time rendering already
auto-scales χ/μ independently (`slices.py::prepare_volume`, one clim per volume). Section A is
**slices-replot-only**.

## Section B — `ROIPickerDialog` (shared widget)

New Qt+matplotlib widget `gui/widgets/roi_picker.py`, imported **lazily** inside the button
handlers (mirrors `line_picker` at `stage_view.py:386`). Imports nothing from `dfxm` beyond
numpy-level types — the widget stays *dumb* (no h5py, no stage imports, no param names).

### Pure helper (testable without Qt)

```python
def rect_to_indices(xmin, xmax, ymin, ymax, w, h) -> tuple[int, int, int, int]:
    """Map a selector rectangle (data coords, pixel-edge extents) to half-open
    (r0, r1, c0, c1) pixel indices, clamped to [0, w]/[0, h]."""
    c0 = max(0, min(int(math.floor(min(xmin, xmax))), w))
    c1 = max(0, min(int(math.ceil(max(xmin, xmax))),  w))
    r0 = max(0, min(int(math.floor(min(ymin, ymax))), h))
    r1 = max(0, min(int(math.ceil(max(ymin, ymax))),  h))
    return r0, r1, c0, c1
```

Half-open + `floor`/`ceil` on **pixel-edge** extents gives inclusive-of-touched-pixels behaviour
with no ±0.5 fencepost. Matches `crop_roi_2d` (figures.py:116) / `apply_roi_3d`
(alignment.py:24–29) semantics: `data[r0:r1, c0:c1]`.

### Dialog

```python
class ROIPickerDialog(QDialog):
    def __init__(self,
                 previews: list[tuple[str, Callable[[], tuple[np.ndarray, float, float]]]],
                 initial: tuple[int, int, int, int] | None = None,
                 parent=None): ...
    result: tuple[int, int, int, int] | None   # (r0, r1, c0, c1) half-open, or None
```

- **Preview dropdown** — one entry per `(label, thunk)`. `thunk() -> (array2d, sx, sy)` is called
  only on first selection of that entry (lazy). A thunk may raise; show the error in an in-dialog
  status label — never crash.
- **Rendering** — `imshow(arr, origin="lower", extent=[0, W, 0, H])` + `ax.set_aspect(sy/sx)`.
  `origin="lower"` matches every export (`render.py:58`, `strain.py:389`) so a picked r-range is
  not vertically mirrored. Physical aspect via `set_aspect(sy/sx)` reproduces the export stretch
  while the selector reports plain pixel-index coords.
- **RectangleSelector** — `interactive=True` (draggable/resizable handles). On each change,
  update the live readout: `{Δr}×{Δc} px  =  {Δr·sy:.1f} × {Δc·sx:.1f} µm (Y×X)`. `Use` disabled
  until a non-degenerate rect (≥1 px each axis) exists.
- **`initial`** — pre-draw the rectangle for the current boxes if all four are set.
- **Preview switch** — selecting a different preview keeps the drawn rectangle **only if the new
  array shape matches**; otherwise clear it and note the shape in the readout (slices groups can
  have different `(nv, nu)`).
- **Buttons** — Reset (clear rect) / Use (accept → set `result`) / Cancel.

## Section C — Preview helpers (Qt-free core)

Each call site supplies previews; the widget stays source-agnostic.

- **Generic replot preview** — a helper beside `_load_layer` in `dfxm/common/figures.py`, e.g.
  `replot_preview(h5_path, key) -> (array2d, sx, sy)`: read the **middle** Z layer of the dataset
  the catalog already exposes; scales from file attrs (`scale_x_um`/`scale_y_um` on strain
  stacks, `scale_x_um_per_px`/`scale_y_um_per_px` on aligned files — cf. slices.py:637–638) with
  a **documented fallback** (the stage's calibrated default) when absent.
- **Slices plane preview** — a small Qt-free helper in `dfxm/stages/slices.py`: read
  `sg["slices"][mid]`; `sx/sy` = median of `np.diff(u_um)` / `np.diff(v_um)` (the resampled slice
  pitch du/dv — **not** detector `sx/sy`).
- **Run-time param previews** — per stage (strain / visualize / slices / paraview), a Qt-free
  `roi_previews(params) -> list[(label, thunk)]` that reads one representative layer from the
  volume/map file named in the current form values. Missing/invalid path → return `[]`
  (button present but the dialog shows "nothing to preview").

## Section D — Wiring

### Replot dialogs (both) — sites P1, P2

- Add a **"Pick ROI…"** button in the ROI row (`replot_dialog.py:75–78`,
  `slice_replot.py:93–96`).
- Each dialog builds its own `previews` from the loaded catalog and passes them to
  `ROIPickerDialog`. To keep `ReplotDialog` module-agnostic, pass a `preview_fn` in alongside the
  existing `catalog_fn`/`render_fn` (constructed in `stage_view.py`). `SliceReplotDialog` uses
  the slices plane-preview helper directly.
- On accept, write `result` ints into the four `QLineEdit`s. Everything downstream is unchanged
  (still four pixel ints → `crop_roi_2d`).
- P1 serves strain / mosaicity / **rocking** *replot* (stored, already-aligned data — safe).

### Run-time stage params — sites R1–R4, schema-driven (no GUI hard-coding)

Add two optional fields to `Param` (`dfxm/config/models.py:52–62`, frozen dataclass — appended
with defaults so every existing spec stays valid):

```python
roi_group: str = ""   # params sharing a roi_group are one picker target
roi_axis: str = ""    # "x" | "y" | "both"  ("both" = a single 4-int "r0,r1,c0,c1" field)
```

`__post_init__` validation: `roi_axis` non-empty ⇒ `roi_group` non-empty and
`roi_axis in {"x","y","both"}`. `tests/test_param_metadata.py` asserts these rules.

Tag the run-time ROI params:

| Stage | Params | Encoding |
|---|---|---|
| strain | `roi` | one field, `roi_axis="both"` → `"r0,r1,c0,c1"` |
| visualize | `roi_x`, `roi_y` | pair, `roi_x`→`"c0,c1"`, `roi_y`→`"r0,r1"` |
| slices | `align_roi_x`, `align_roi_y` | pair, same encoding |
| paraview | `roi_x`, `roi_y` | pair, same encoding |
| rocking | `roi_x`, `roi_y` | **not tagged** — excluded (raw-frame) |

Each of the four stages exposes `roi_previews(params)` (Section C). `StageView` (not
`ParamForm`) scans the active spec for `roi_group`s, appends a **"Pick ROI…"** button next to
those rows, opens `ROIPickerDialog` with the stage's previews, and maps `result` back per
`roi_axis`:
- `both` → set the one field to `"{r0},{r1},{c0},{c1}"`.
- `x` → set the x-param to `"{c0},{c1}"`; `y` → set the y-param to `"{r0},{r1}"` (both members of
  the roi_group updated together from the single picked rectangle).

Write-back uses `self._form.set_values(...)` — the same mechanism the line picker uses to inject
`jobs_json` (`stage_view.py:400–409`). **Per-stage only** — no writes to other stages' forms.

## Testing

- **Pure:** `rect_to_indices` — edges, degenerate (zero-area), clamping beyond bounds, swapped
  min/max. No Qt.
- **Core:** slices `render_replot` on a synthetic `oblique_slices.h5` — a `{volume_id: (vmin,vmax)}`
  dict gives χ and μ *different* applied limits; a group-keyed dict still applies via the
  fallback; a legacy single tuple still applies to all. Preview helpers return correct
  `(array, sx, sy)` on synthetic files (incl. du/dv for slices).
- **GUI smoke** (`tests/gui_smoke.py`, offscreen): construct `ROIPickerDialog` with synthetic
  thunks, drive the selector callback directly with fake extents (do **not** synthesise Qt mouse
  drags), assert `result` and the readout text; assert `SliceReplotDialog` emits one clim row per
  volume_id; assert a `StageView` on a roi-grouped spec shows a "Pick ROI…" button that writes
  the expected encoded strings.
- **`tests/test_param_metadata.py`:** `roi_group`/`roi_axis` validity rules.

## Documentation (same-change contract)

- **`docs/Usage.md`** — slices replot: per-quantity colour limits (χ/μ and raw split); the
  "Pick ROI…" picker in both replot dialogs and on the strain/visualize/slices/paraview forms;
  a one-line note that "the preview is oriented exactly like the exported maps" (origin-lower).
- **`docs/Codebase.md`** — new `gui/widgets/roi_picker.py` (`rect_to_indices`, `ROIPickerDialog`);
  the `Param.roi_group`/`roi_axis` fields; the `replot_preview`/slices-plane-preview/per-stage
  `roi_previews` helpers; the changed slices `render_replot` clim key.

## Risks & correctness traps (from Fable review)

1. **Origin.** Exports are `origin="lower"`; the picker must match or every picked r-range is
   mirrored. Highest-severity trap — cover with a `rect_to_indices` test + a smoke assertion that
   a rectangle over a known bright block recovers that block's row indices. Add a Usage.md
   sentence for darfix users (image origin top-left).
2. **Half-open / pixel edges.** Use `extent=[0,W,0,H]` + floor/ceil, *not* the no-extent imshow
   default (pixel centres at integers, edges at −0.5) — that is where off-by-ones breed.
3. **Slices pitch ≠ detector pitch.** Slice planes are resampled at du/dv; the slices adapter
   must read du/dv from the file, never global sx/sy or the form.
4. **One ROI, many shapes (slices).** `render_replot` applies one roi to every selected plane;
   different groups can differ in `(nv, nu)`. Clamping already makes this safe; the picker shows
   only one plane, so clear the rect on shape change and keep the tree's `{Y}×{X} px` labels.
5. **Strain frame subtleties.** Replot ROI indices are in the *stored* (already run-ROI-cropped)
   layer frame; preview the stored layer so it stays self-consistent. Do not promise
   absolute-µm positions in the readout.
6. **Clim key migration.** The two-key fallback (Section A) is mandatory; without it any caller
   still passing group-keyed dicts silently loses overrides.
7. **Over-coupling.** The widget must import nothing from `dfxm` except numpy-level types. If the
   schema fields start needing per-stage special cases in `param_form.py`, stop and re-plan.
8. **Laziness.** Import `roi_picker` (and matplotlib's Qt backend) only inside the button
   handlers.

## Out of scope

- Cross-stage crop propagation (per-stage only).
- Rocking run-time ROI picker (manual boxes retained).
- silx adoption.
- Any change to the run-time ROI *semantics* — the picker only *produces* the same ints users
  type today.
