# Figure builder: trace y-axis numbers + external image panels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a trace panel drop its y-axis numbers per panel, and let a PNG/JPEG/TIFF file join a composed figure as a lettered `"image"` panel sized by printed width.

**Architecture:** Both features are additive fields on the Qt-free recipe model (`dfxm/compose/recipe.py`, `RECIPE_VERSION` stays 1). The y-axis toggle is one `PanelDef` bool consumed in `adapters.draw_panel`'s trace branch. The image panel is a fifth `PANEL_KINDS` entry with its own loader, layout cell, and draw branch; because it is an ordinary `PanelRef`, lettering, the drag arranger, overrides and relative-path save/load work without special-casing. The GUI (`gui/figure_builder.py`) gains one checkbox, one spin box, an "Add image…" button, and a per-kind widget-enable rule.

**Tech Stack:** Python 3, matplotlib (`matplotlib.image.imread`/`imsave` — Pillow arrives with matplotlib, so **no new dependency**), PySide6 (GUI only), pytest with `QT_QPA_PLATFORM=offscreen`.

**Spec:** `docs/superpowers/specs/2026-09-02-figure-builder-ynumbers-image-design.md`

## Global Constraints

- `dfxm/` stays Qt-free; matplotlib is imported inside functions in `adapters.py` (existing pattern).
- `RECIPE_VERSION = 1` is not bumped; every new JSON key is additive with a default on load.
- No new entry in `pyproject.toml` or the two docs dependency lists.
- Every task that changes `dfxm/stages/`, `dfxm/compose/` behaviour or `gui/` updates `docs/Usage.md` and/or `docs/Codebase.md` **in the same commit** (CLAUDE.md docs contract).
- Run the suite as `python3 -m pytest -q --ignore=tests/test_gui_viewer3d.py` (that file cannot run under pytest on this machine — report it NOT RUN). Never run `tests/gui_smoke.py` and pytest at once.
- `ruff format` runs on every Write/Edit via hook; run `ruff check .` before each commit.
- Commit message trailer (every commit):
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01CxfuD3zbfpfRAjnrK37t1r
  ```
- Never reconstruct an `old_string` from memory: Read the target lines right before each Edit (line numbers below are from baseline `db98c5e` and shift as tasks land).

## File map

| File | Responsibility in this plan |
|---|---|
| `dfxm/compose/recipe.py` | schema: `PanelDef.y_tick_labels`, `PanelDef.width_cm`, `"image"` in `PANEL_KINDS`, `IMAGE_DEFAULT_WIDTH_CM`, dict helpers, validation |
| `dfxm/compose/adapters.py` | `_load_image` loader, `draw_panel` image + y-tick branches, `panel_preview` refusal |
| `dfxm/compose/layout.py` | `_image_cell` sizing, `SizedCell.kind == "image"` |
| `dfxm/compose/render.py` | draw-loop comment only (image goes through the existing else branch) |
| `dfxm/compose/gridmap.py` | `panel_group_hint` returns `None` for `"image"` |
| `gui/figure_builder.py` | checkbox, width spin, Add image… button, `add_image()`, per-kind enable rule, `_IMAGE_KINDS` |
| `tests/test_compose_recipe.py`, `tests/test_compose_adapters.py`, `tests/test_compose_layout.py`, `tests/test_compose_render.py`, `tests/test_compose_cli.py`, `tests/test_compose_gridmap.py`, `tests/test_gui_figure_builder.py` | one test file per module touched |
| `docs/Usage.md` §Figure builder, `docs/Codebase.md` §`dfxm/compose` + `figure_builder.py` row | docs contract |

---

### Task 1: `PanelDef.y_tick_labels` in the recipe schema

**Files:**
- Modify: `dfxm/compose/recipe.py:47-61` (dataclass), `:250-281` (dict helpers)
- Modify: `docs/Codebase.md:1092-1102` (`PanelDef` bullet)
- Test: `tests/test_compose_recipe.py`

**Interfaces:**
- Produces: `PanelDef.y_tick_labels: bool = True`; JSON key `"y_tick_labels"` (absent → `True`).

- [ ] **Step 1: Write the failing test** — append to `tests/test_compose_recipe.py`:

```python
def test_y_tick_labels_round_trips_and_old_recipe_defaults_true():
    import json

    r = _mini_recipe()
    r.panels[1].y_tick_labels = False  # the trace panel
    r2 = recipe_from_json(recipe_to_json(r))
    assert r2.panels[1].y_tick_labels is False and r2.panels[0].y_tick_labels is True
    d = json.loads(recipe_to_json(_mini_recipe()))
    for p in d["panels"]:
        p.pop("y_tick_labels")  # a recipe written before the field existed
    r3 = recipe_from_json(json.dumps(d))
    assert all(p.y_tick_labels is True for p in r3.panels)
```

- [ ] **Step 2: Run it, expect failure**

Run: `python3 -m pytest tests/test_compose_recipe.py::test_y_tick_labels_round_trips_and_old_recipe_defaults_true -q`
Expected: FAIL — `KeyError: 'y_tick_labels'` (the pop) or `AttributeError`.

- [ ] **Step 3: Add the field and the two dict lines**

In `PanelDef`, after `crop_to_data: bool = False`:

```python
    # Trace panels only: False hides the y tick labels and the ×10ⁿ offset
    # text (tick marks, grid and the y-label stay); ignored by other kinds.
    y_tick_labels: bool = True
```

In `_panel_def_to_dict`, after `"crop_to_data": bool(p.crop_to_data),`:

```python
        "y_tick_labels": bool(p.y_tick_labels),
```

In `_panel_def_from_dict`, after `crop_to_data=bool(d.get("crop_to_data", False)),`:

```python
        y_tick_labels=bool(d.get("y_tick_labels", True)),
```

- [ ] **Step 4: Run the recipe tests**

Run: `python3 -m pytest tests/test_compose_recipe.py -q`
Expected: all PASS.

- [ ] **Step 5: Docs** — in `docs/Codebase.md` `PanelDef` bullet (≈line 1092), extend the override list `(`roi`, `clim`, … `crop_to_data`)` to end `…, `crop_to_data`, `y_tick_labels`)` and append one sentence after the `crop_to_data` explanation:

```
`y_tick_labels` (bool, default `True`, 2026-09-02) applies to `profiles_trace`
panels only: `False` makes `adapters.draw_panel` hide the y tick labels and
the y offset text (tick marks, grid and y-label stay); other kinds ignore it;
additive in JSON (`bool(d.get("y_tick_labels", True))`).
```

- [ ] **Step 6: Commit**

```bash
ruff check dfxm/compose/recipe.py tests/test_compose_recipe.py
git add dfxm/compose/recipe.py tests/test_compose_recipe.py docs/Codebase.md
git commit -m "feat(compose): PanelDef.y_tick_labels (additive, default True)"
```

---

### Task 2: Hide a trace panel's y numbers in `draw_panel`

**Files:**
- Modify: `dfxm/compose/adapters.py:442-464` (trace branch of `draw_panel`)
- Modify: `docs/Codebase.md:1259` (`draw_panel` bullet)
- Test: `tests/test_compose_adapters.py`

**Interfaces:**
- Consumes: `PanelDef.y_tick_labels` (Task 1).
- Produces: nothing new; behaviour only.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_compose_adapters.py`:

```python
def test_draw_panel_trace_y_tick_labels_off_hides_numbers_keeps_label(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    p = PanelDef(
        "t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}), y_tick_labels=False
    )
    draw_panel(ax, p, load_panel(p), None)
    assert ax.get_yticklabels() == []  # matplotlib drops invisible labels from this list
    assert ax.yaxis.get_offset_text().get_visible() is False
    assert ax.get_ylabel() and ax.yaxis.label.get_visible()
    assert ax.yaxis.get_major_ticks()[0].tick1line.get_visible()  # tick marks stay
    assert ax.get_xticklabels()  # x numbers untouched


def test_draw_panel_y_tick_labels_ignored_by_non_trace_kinds(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    sel = {"volume_id": "strain", "slice_name": "obl", "plane": 0}
    p = PanelDef("s", PanelSource(h5, "slice_plane", sel), y_tick_labels=False)
    draw_panel(ax, p, load_panel(p), None, colorbar=False, scale_bar=False)
    assert ax.yaxis.get_offset_text().get_visible() is True
```

- [ ] **Step 2: Run them, expect the first to fail**

Run: `python3 -m pytest tests/test_compose_adapters.py -q -k y_tick_labels`
Expected: first FAILS on `ax.get_yticklabels() == []`; second PASSES already.

- [ ] **Step 3: Implement** — in `draw_panel`'s `profiles_trace` branch, replace

```python
        if not show_xlabel:
            ax.tick_params(labelbottom=False)
        return None
```

with

```python
        if not show_xlabel:
            ax.tick_params(labelbottom=False)
        if not panel.y_tick_labels:
            # numbers only: tick marks, grid and the y-label stay
            ax.tick_params(labelleft=False)
            ax.yaxis.get_offset_text().set_visible(False)
        return None
```

- [ ] **Step 4: Run the adapters tests**

Run: `python3 -m pytest tests/test_compose_adapters.py -q`
Expected: all PASS.

- [ ] **Step 5: Docs** — `docs/Codebase.md` `draw_panel` bullet (≈line 1259): after the sentence describing the trace branch's `show_xlabel` handling, add: "When `panel.y_tick_labels` is `False` the trace branch also calls `ax.tick_params(labelleft=False)` and hides `ax.yaxis.get_offset_text()` — the tick marks, grid and y-label are untouched, and `layout.measure_cells` shrinks the left margin on its own because it measures what was drawn."

- [ ] **Step 6: Commit**

```bash
ruff check dfxm/compose/adapters.py tests/test_compose_adapters.py
git add dfxm/compose/adapters.py tests/test_compose_adapters.py docs/Codebase.md
git commit -m "feat(compose): y_tick_labels=False hides a trace panel's y numbers"
```

---

### Task 3: "Y-axis numbers" checkbox in the panel inspector + Usage docs

**Files:**
- Modify: `gui/figure_builder.py:964-971` (after the Crop-to-data checkbox), `:1037-1074` (`_load_panel_page`), `:1189-1198` (getters), `:1270-1287` (`changes`)
- Modify: `docs/Usage.md:2494-2525` (Panel page bullet), `docs/Codebase.md:1931` (`figure_builder.py` row, Panel page sentence)
- Test: `tests/test_gui_figure_builder.py`

**Interfaces:**
- Produces: `FigureBuilderWindow._ov_ynums: QCheckBox`; override key `"y_tick_labels"`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_gui_figure_builder.py`:

```python
def test_override_y_tick_labels_checkbox_trace_only():
    import json

    from dfxm.compose.recipe import recipe_to_json

    w = _win()
    trace = PanelDef("t", PanelSource("/x.h5", "profiles_trace", {"job": {}, "field": "strain"}))
    w.add_panels([_panel("a"), trace])
    w._select_outline_panel("t")
    assert w._ov_ynums.isEnabled() and w._ov_ynums.isChecked()
    w._ov_ynums.setChecked(False)  # real widget signal path
    assert trace.y_tick_labels is False and w.is_dirty()
    assert json.loads(recipe_to_json(w.recipe()))["panels"][1]["y_tick_labels"] is False
    trace.y_tick_labels = True
    w._load_panel_page(trace)
    assert w._ov_ynums.isChecked()
    w._select_outline_panel("a")
    assert not w._ov_ynums.isEnabled()  # map panels have no such switch
```

- [ ] **Step 2: Run it, expect failure**

Run: `python3 -m pytest tests/test_gui_figure_builder.py::test_override_y_tick_labels_checkbox_trace_only -q`
Expected: FAIL — `AttributeError: ... has no attribute '_ov_ynums'`.

- [ ] **Step 3: Add the widget** — in `_build_override_editor`, right after `form.addRow("", self._ov_crop)`:

```python
        self._ov_ynums = QCheckBox("Y-axis numbers")
        self._ov_ynums.setToolTip(
            "Untick to drop the tick numbers (and any ×10ⁿ offset) from this trace's "
            "y-axis; the tick marks and the y-label stay. Trace panels only."
        )
        self._ov_ynums.toggled.connect(lambda _c: self._on_override_field_edited("y_tick_labels"))
        form.addRow("", self._ov_ynums)
```

- [ ] **Step 4: Wire getter, changes, load** —

In `_on_override_field_edited`'s `getters` dict add `"y_tick_labels": self._ov_ynums.isChecked,` after the `"crop_to_data"` entry.

In `_apply_panel_overrides`, after the `if "crop_to_data" in values:` block add:

```python
        if "y_tick_labels" in values:
            changes["y_tick_labels"] = bool(values["y_tick_labels"])
```

In `_load_panel_page`: add `self._ov_ynums,` to the `widgets` tuple (after `self._ov_crop,`); after `self._ov_crop.setChecked(bool(panel.crop_to_data))` add:

```python
        self._ov_ynums.setChecked(bool(panel.y_tick_labels))
        self._ov_ynums.setEnabled(panel.source.kind == "profiles_trace")
```

(Task 8 replaces that `setEnabled` line with the full per-kind rule.)

- [ ] **Step 5: Run the builder tests**

Run: `python3 -m pytest tests/test_gui_figure_builder.py -q`
Expected: all PASS.

- [ ] **Step 6: Docs** —

`docs/Usage.md` Panel bullet (≈2494): after the Crop-to-data clause "…trace panels ignore it)," insert:

```
    a **Y-axis numbers** checkbox (trace panels only — untick it to print the
    line-shape's sum-intensity axis without tick numbers or a ×10ⁿ offset; the
    tick marks, the grid and the y-label stay, and the panel's left margin
    shrinks to match; greyed out on map panels),
```

`docs/Codebase.md` `figure_builder.py` row: in the **Panel page** sentence, after the `self._ov_crop` clause add "`self._ov_ynums` (`QCheckBox` "Y-axis numbers", 2026-09-02, bound to `PanelDef.y_tick_labels` through `_on_override_field_edited("y_tick_labels")`; enabled only for `profiles_trace` panels)", and add `"y_tick_labels"` to the list of fixed field names in the `_on_override_field_edited(key)` clause.

- [ ] **Step 7: Full suite + commit**

Run: `python3 -m pytest -q --ignore=tests/test_gui_viewer3d.py && ruff check .`
Expected: all PASS, no ruff findings.

```bash
git add gui/figure_builder.py tests/test_gui_figure_builder.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): per-panel Y-axis numbers switch for trace panels"
```

---

### Task 4: `"image"` panel kind + `PanelDef.width_cm` in the schema

**Files:**
- Modify: `dfxm/compose/recipe.py:12` (`PANEL_KINDS`), `:40-44` (`PanelSource` comment), `:47-61` (`PanelDef`), `:250-281` (dict helpers), `:451-462` (validation tail)
- Modify: `docs/Codebase.md:1068`, `1090-1102`, `1149-1155`
- Test: `tests/test_compose_recipe.py`

**Interfaces:**
- Produces: `PANEL_KINDS` includes `"image"`; `IMAGE_DEFAULT_WIDTH_CM = 6.0`; `PanelDef.width_cm: float | None = None`; JSON key `"width_cm"`; `validate_recipe` rejects `width_cm <= 0`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_compose_recipe.py`:

```python
def test_image_panel_width_round_trips_relative_path_and_defaults():
    import json
    import os

    r = _mini_recipe()
    r.panels.append(PanelDef("i0", __src("/data/figs/ref.png", "image", {}), width_cm=4.5))
    r.layout.children.append(PanelRef("i0"))
    txt = recipe_to_json(r, base_dir="/data")
    d = json.loads(txt)
    assert d["panels"][2]["source"]["h5_path"] == os.path.join("figs", "ref.png")
    r2 = recipe_from_json(txt, base_dir="/data")
    img = r2.panels[2]
    assert img.source.kind == "image" and img.source.h5_path == "/data/figs/ref.png"
    assert img.width_cm == 4.5 and r2.panels[0].width_cm is None
    validate_recipe(r2)  # "image" is a legal kind
    for p in d["panels"]:
        p.pop("width_cm")
    r3 = recipe_from_json(json.dumps(d), base_dir="/data")
    assert all(p.width_cm is None for p in r3.panels)


def test_image_panel_width_validated():
    r = _mini_recipe()
    r.panels.append(PanelDef("i0", __src("/data/ref.png", "image", {}), width_cm=0.0))
    r.layout.children.append(PanelRef("i0"))
    with pytest.raises(StageUserError) as e:
        validate_recipe(r)
    assert "width_cm" in str(e.value) and e.value.hint
```

- [ ] **Step 2: Run them, expect failure**

Run: `python3 -m pytest tests/test_compose_recipe.py -q -k image_panel`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'width_cm'`.

- [ ] **Step 3: Schema edits** —

Line 12:

```python
PANEL_KINDS = ("map_layer", "slice_plane", "profiles_ref", "profiles_trace", "image")
IMAGE_DEFAULT_WIDTH_CM = 6.0  # printed width of an "image" panel when PanelDef.width_cm is None
```

`PanelSource.h5_path` line:

```python
    h5_path: str  # source file: an .h5 for the data kinds, a PNG/JPEG/TIFF for "image"
```

`PanelDef`, after the `y_tick_labels` field:

```python
    # Image panels only: printed width in cm; None = IMAGE_DEFAULT_WIDTH_CM.
    # Height always follows the image's pixel aspect (never a µm/cm scale).
    width_cm: float | None = None
```

`_panel_def_to_dict`: add `"width_cm": p.width_cm,` after the `"y_tick_labels"` line.
`_panel_def_from_dict`: add `width_cm=d.get("width_cm"),` after the `y_tick_labels=` line.

`validate_recipe`, appended at the very end (after the `compose.{field}` loop):

```python
    for p in recipe.panels:
        if p.width_cm is not None and not (float(p.width_cm) > 0):
            raise StageUserError(
                f"panel {p.id!r}: width_cm must be positive, got {p.width_cm!r}",
                hint="Set the image panel's Width (cm) to a positive number, or leave it at "
                "the default.",
            )
```

- [ ] **Step 4: Run the recipe tests**

Run: `python3 -m pytest tests/test_compose_recipe.py -q`
Expected: all PASS (including `test_validate_refuses_bad_recipes`, whose "unknown kind" case must still use a kind that is not in the tuple — check it does not use `"image"`).

- [ ] **Step 5: Docs** — `docs/Codebase.md`:
  - line ≈1068: update the `PANEL_KINDS` tuple to include `"image"` and add `IMAGE_DEFAULT_WIDTH_CM = 6.0`.
  - `PanelSource` bullet (≈1090): `h5_path` is "the source file path — an `.h5` for the four data kinds, a PNG/JPEG/TIFF for `"image"` (2026-09-02; the JSON key keeps its historical name so v1 recipes stay readable, and `_rel_path`/`_resolve_path` relativise it on save/load whatever the kind)".
  - `PanelDef` bullet: add `width_cm` to the override list and the sentence "`width_cm` (`float | None`, image panels only, 2026-09-02): printed width in cm, `None` = `IMAGE_DEFAULT_WIDTH_CM`; height follows the pixel aspect in `layout._image_cell`; `validate_recipe` rejects a non-positive value (`StageUserError` + hint); additive in JSON (`d.get("width_cm")`)."
  - `validate_recipe` bullet (≈1149): append "a non-positive `PanelDef.width_cm`" to the list of refusals.

- [ ] **Step 6: Commit**

```bash
ruff check dfxm/compose/recipe.py tests/test_compose_recipe.py
git add dfxm/compose/recipe.py tests/test_compose_recipe.py docs/Codebase.md
git commit -m "feat(compose): 'image' panel kind + PanelDef.width_cm in the recipe schema"
```

---

### Task 5: `_load_image` loader + `panel_preview` refusal

**Files:**
- Modify: `dfxm/compose/adapters.py:11-23` (module docstring selector list), `:468-476` (`panel_preview`), `:518-523` (`_LOADERS`), new function above `_LOADERS`
- Modify: `docs/Codebase.md:1206-1226` (`load_panel` bullet), `:1247` (selector list), `:1270` (`panel_preview` bullet)
- Test: `tests/test_compose_adapters.py`

**Interfaces:**
- Consumes: `crop_roi_2d(layer, roi)` from `dfxm.common.figures` (already imported in `adapters.py`; slices the first two axes, returns `None` for an empty crop, so it works on an `(h, w, 3)` array).
- Produces: `_load_image(path, sel, roi, *, crop_to_data=False) -> PanelData` with `kind="image"`, `ext_x_um=float(w_px)`, `ext_y_um=float(h_px)`, `group=None`, `payload={"image": ndarray}` (float in 0–1, shape `(h, w)`, `(h, w, 3)` or `(h, w, 4)`). Test helper `_write_png(path, w=40, h=20) -> str` (reused by Tasks 7 and 8 — copy it, do not import across test files).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_compose_adapters.py`:

```python
def _write_png(path, w=40, h=20):
    from matplotlib.image import imsave

    rgb = np.zeros((h, w, 3), "f4")
    rgb[..., 0] = np.linspace(0.0, 1.0, w)[None, :]
    imsave(str(path), rgb)
    return str(path)


def test_load_image_pixels_as_extent_and_float_rgb_payload(tmp_path):
    png = _write_png(tmp_path / "ref.png")
    d = load_panel(PanelDef("i", PanelSource(png, "image", {})))
    assert d.kind == "image" and (d.ext_x_um, d.ext_y_um) == (40.0, 20.0)
    img = d.payload["image"]
    assert img.shape[:2] == (20, 40) and img.dtype.kind == "f"
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert d.group is None and d.vmin is None


def test_load_image_roi_is_a_pixel_crop_and_empty_crop_is_placeholder(tmp_path):
    png = _write_png(tmp_path / "ref.png")
    d = load_panel(PanelDef("i", PanelSource(png, "image", {}), roi=(5, 15, 10, 30)))
    assert (d.ext_x_um, d.ext_y_um) == (20.0, 10.0)
    assert d.payload["image"].shape[:2] == (10, 20)
    d2 = load_panel(PanelDef("i", PanelSource(png, "image", {}), roi=(5, 5, 10, 30)))
    assert d2.kind == "placeholder" and "ref.png" in d2.payload["reason"]


def test_load_image_missing_file_is_placeholder_not_error(tmp_path):
    d = load_panel(PanelDef("i", PanelSource(str(tmp_path / "gone.png"), "image", {})))
    assert d.kind == "placeholder" and "gone.png" in d.payload["reason"]


def test_load_image_crop_to_data_is_ignored(tmp_path):
    png = _write_png(tmp_path / "ref.png")
    d = load_panel(PanelDef("i", PanelSource(png, "image", {}), crop_to_data=True))
    assert (d.ext_x_um, d.ext_y_um) == (40.0, 20.0)


def test_panel_preview_refuses_image_panel(tmp_path):
    from dfxm.compose.adapters import panel_preview

    png = _write_png(tmp_path / "ref.png")
    with pytest.raises(ValueError, match="pixel crop"):
        panel_preview(PanelDef("i", PanelSource(png, "image", {})))
```

- [ ] **Step 2: Run them, expect failure**

Run: `python3 -m pytest tests/test_compose_adapters.py -q -k "load_image or preview_refuses_image"`
Expected: FAIL — `KeyError: 'image'` from `_LOADERS[...]` (the kind now passes the `PANEL_KINDS` check).

- [ ] **Step 3: Implement the loader** — insert directly above `_LOADERS = {`:

```python
def _load_image(path, sel, roi, *, crop_to_data=False) -> PanelData:
    """An external raster (PNG/JPEG/TIFF) as a lettered panel.

    ``ext_x_um``/``ext_y_um`` carry PIXELS — used only for the aspect ratio by
    ``layout._image_cell``, never as a physical scale. ``roi`` is a plain
    ``(r0, r1, c0, c1)`` pixel crop; ``crop_to_data`` is ignored. A missing or
    undecodable file raises and becomes a placeholder in ``load_panel``.
    """
    import numpy as np
    from matplotlib.image import imread

    arr = np.asarray(imread(path))
    if arr.dtype.kind in "ui":
        arr = arr.astype("f4") / float(np.iinfo(arr.dtype).max)
    cropped = crop_roi_2d(arr, roi)
    if cropped is None or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        raise ValueError(f"ROI {tuple(roi)} leaves no pixels")
    h, w = cropped.shape[:2]
    return PanelData(kind="image", ext_x_um=float(w), ext_y_um=float(h), payload={"image": cropped})
```

Add `"image": _load_image,` as the last `_LOADERS` entry.

In `panel_preview`, after the `profiles_trace` refusal:

```python
    if panel.source.kind == "image":
        raise ValueError("an image panel's ROI is a pixel crop — type it in the ROI box")
```

Module docstring selector list: add `- ``image``: ``{}`` — no selector; ``h5_path`` is the PNG/JPEG/TIFF file itself.`

- [ ] **Step 4: Run the adapters tests**

Run: `python3 -m pytest tests/test_compose_adapters.py -q`
Expected: all PASS.

- [ ] **Step 5: Docs** — `docs/Codebase.md`:
  - `load_panel` bullet: after "(the trace loader accepts and ignores the flag)" add "; the `image` loader (`_load_image`, 2026-09-02) reads the file with `matplotlib.image.imread` (PNG natively, JPEG/TIFF via the Pillow that matplotlib already depends on — no new dependency), normalises integer arrays to float 0–1, applies `roi` as a plain `crop_roi_2d` pixel crop, ignores `crop_to_data`, and returns `ext_x_um`/`ext_y_um` in **pixels** (aspect only), `group=None`, `payload={"image": array}`; a missing/unreadable file or an empty crop is a placeholder like any other data failure".
  - selector list: add `- `image`: `{}` — no selector; `h5_path` is the image file.`
  - `panel_preview` bullet: add "and `ValueError("…pixel crop…")` for `image` panels (the picker has no preview for them; the ROI text box still works)".

- [ ] **Step 6: Commit**

```bash
ruff check dfxm/compose/adapters.py tests/test_compose_adapters.py
git add dfxm/compose/adapters.py tests/test_compose_adapters.py docs/Codebase.md
git commit -m "feat(compose): image loader — PNG/JPEG/TIFF as PanelData kind 'image'"
```

---

### Task 6: `_image_cell` sizing in the layout pass

**Files:**
- Modify: `dfxm/compose/layout.py:26` (import), `:67-71` (`SizedCell.kind` comment), `:118-138` (`leaf_cell`), new `_image_cell` after `_map_cell`
- Modify: `docs/Codebase.md:1297-1300` (`SizedCell`), `:1308-1330` (`size_cells` rules)
- Test: `tests/test_compose_layout.py`

**Interfaces:**
- Consumes: `IMAGE_DEFAULT_WIDTH_CM` (Task 4), `PanelData(kind="image", ext_x_um=w_px, ext_y_um=h_px)` (Task 5).
- Produces: `SizedCell.kind == "image"`; sizing rule: pinned row height wins (width from aspect), else pinned column width (height from aspect), else `panel.width_cm or IMAGE_DEFAULT_WIDTH_CM`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_compose_layout.py` (uses the file's existing `_panel`, `_recipe`, `_trace_data` helpers):

```python
def _image_data(w=40.0, h=20.0):
    return PanelData(kind="image", ext_x_um=w, ext_y_um=h)


def test_image_cell_default_width_and_aspect_no_scale_needed():
    layout = PanelRef("i")
    cells = size_cells(_recipe(layout, [_panel("i", "image")]), PlotStyle(), {"i": _image_data()}, notes := [])
    c = cells[id(layout)]
    assert c.kind == "image"
    assert abs(c.w_in - 6.0 / 2.54) < 1e-9 and abs(c.h_in - 3.0 / 2.54) < 1e-9
    assert notes == []


def test_image_cell_width_cm_override():
    p = _panel("i", "image")
    p.width_cm = 3.0
    layout = PanelRef("i")
    c = size_cells(_recipe(layout, [p]), PlotStyle(), {"i": _image_data()}, [])[id(layout)]
    assert abs(c.w_in - 3.0 / 2.54) < 1e-9 and abs(c.h_in - 1.5 / 2.54) < 1e-9


def test_image_cell_pinned_row_height_sets_width_from_aspect():
    layout = Row([PanelRef("i")], pinned_height_cm=4.0)
    cells = size_cells(_recipe(layout, [_panel("i", "image")]), PlotStyle(), {"i": _image_data()}, notes := [])
    c = cells[id(layout.children[0])]
    assert abs(c.h_in - 4.0 / 2.54) < 1e-9 and abs(c.w_in - 8.0 / 2.54) < 1e-9
    assert any("pinned row height" in n for n in notes)


def test_image_cell_pinned_col_width_sets_height_from_aspect():
    layout = Col([PanelRef("i")], pinned_width_cm=2.0)
    c = size_cells(_recipe(layout, [_panel("i", "image")]), PlotStyle(), {"i": _image_data()}, [])[
        id(layout.children[0])
    ]
    assert abs(c.w_in - 2.0 / 2.54) < 1e-9 and abs(c.h_in - 1.0 / 2.54) < 1e-9


def test_image_cell_double_pin_height_wins_with_note():
    layout = Row([Col([PanelRef("i")], pinned_width_cm=1.0)], pinned_height_cm=4.0)
    cells = size_cells(_recipe(layout, [_panel("i", "image")]), PlotStyle(), {"i": _image_data()}, notes := [])
    c = cells[id(layout.children[0].children[0])]
    assert abs(c.h_in - 4.0 / 2.54) < 1e-9 and abs(c.w_in - 8.0 / 2.54) < 1e-9
    assert any("height pin wins" in n for n in notes)


def test_image_cell_degenerate_pixels_is_placeholder():
    layout = PanelRef("i")
    cells = size_cells(_recipe(layout, [_panel("i", "image")]), PlotStyle(), {"i": _image_data(0.0, 5.0)}, notes := [])
    assert cells[id(layout)].kind == "placeholder" and any("degenerate image" in n for n in notes)


def test_image_cell_is_never_a_trace_autoscale_target():
    style = PlotStyle(trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    layout = Col([PanelRef("i"), PanelRef("t")])
    r = _recipe(layout, [_panel("i", "image"), _panel("t", "profiles_trace")])
    cells = size_cells(r, style, {"i": _image_data(), "t": _trace_data()}, [])
    assert trace_column_targets(r, cells)[id(layout.children[1])] is None
```

- [ ] **Step 2: Run them, expect failure**

Run: `python3 -m pytest tests/test_compose_layout.py -q -k image_cell`
Expected: FAIL — the first raises `StageUserError` "has no physical scale to size from" (image fell through to `_map_cell`).

- [ ] **Step 3: Implement** —

Import line 26 becomes:

```python
from .recipe import (
    IMAGE_DEFAULT_WIDTH_CM,
    Col,
    PanelRef,
    Row,
    ScaleBarCell,
    Spacer,
    TextCell,
    iter_leaves,
)
```

`SizedCell.kind` comment: `kind: str  # "map"|"trace"|"image"|"spacer"|"text"|"scalebar"|"placeholder"`.

In `leaf_cell`, before `if data.kind == "profiles_trace":`:

```python
        if data.kind == "image":
            return _image_cell(leaf, panel, data, pinned_h_in, pinned_w_in)
```

New function after `_map_cell` (inside `size_cells`, same indentation as `_map_cell`):

```python
    def _image_cell(leaf, panel, data, pinned_h_in, pinned_w_in):
        # ext_* are PIXELS here — only the aspect matters; no µm/cm scale is
        # ever consulted for an image.
        w_px, h_px = data.ext_x_um, data.ext_y_um
        if not (_finite_positive(w_px) and _finite_positive(h_px)):
            notes.append(f"panel {panel.id}: degenerate image size — rendered as placeholder")
            return SizedCell(
                leaf,
                panel,
                "placeholder",
                PLACEHOLDER_CM[0] * _IN_PER_CM,
                PLACEHOLDER_CM[1] * _IN_PER_CM,
            )
        if pinned_h_in is not None:
            h = pinned_h_in
            w = h * w_px / h_px
            notes.append(
                f"panel {panel.id}: pinned row height — image width {w / _IN_PER_CM:.4g} cm "
                "follows its aspect"
            )
            if pinned_w_in is not None:
                notes.append(
                    f"panel {panel.id}: both row height and column width pinned — "
                    "height pin wins (image aspect is fixed); width pin ignored"
                )
            return SizedCell(leaf, panel, "image", w, h)
        if pinned_w_in is not None:
            w = pinned_w_in
            return SizedCell(leaf, panel, "image", w, w * h_px / w_px)
        w = (panel.width_cm or IMAGE_DEFAULT_WIDTH_CM) * _IN_PER_CM
        return SizedCell(leaf, panel, "image", w, w * h_px / w_px)
```

- [ ] **Step 4: Run the layout tests**

Run: `python3 -m pytest tests/test_compose_layout.py -q`
Expected: all PASS.

- [ ] **Step 5: Docs** — `docs/Codebase.md`:
  - `SizedCell` bullet: add `"image"` to the `kind` list.
  - `size_cells` rules list: add a bullet "`PanelData(kind="image")` (2026-09-02, `_image_cell`): box from **pixels**, never a scale — a pinned row height sets the height and the width follows the pixel aspect (note; with a column width also pinned, the height pin wins, note); else a pinned column width sets the width and the height follows; else `panel.width_cm or IMAGE_DEFAULT_WIDTH_CM` wide. Degenerate pixel dimensions → `PLACEHOLDER_CM` + note. `trace_column_targets.widest_map` keys on `kind == "map"`, so an image is never a trace-autoscale target."

- [ ] **Step 6: Commit**

```bash
ruff check dfxm/compose/layout.py tests/test_compose_layout.py
git add dfxm/compose/layout.py tests/test_compose_layout.py docs/Codebase.md
git commit -m "feat(compose): size image panels by printed width and pixel aspect"
```

---

### Task 7: Draw the image, render end to end, arranger chip, CLI

**Files:**
- Modify: `dfxm/compose/adapters.py:347-349` (`draw_panel`, after the placeholder branch)
- Modify: `dfxm/compose/render.py:971-972` (draw-loop else comment)
- Modify: `dfxm/compose/gridmap.py:98-113` (`panel_group_hint`)
- Modify: `docs/Codebase.md:1259` (`draw_panel`), `:1183` (`panel_group_hint`), render draw-loop paragraph (grep `else:  # placeholder` / "placeholder" near line 1659)
- Test: `tests/test_compose_adapters.py`, `tests/test_compose_render.py`, `tests/test_compose_gridmap.py`, `tests/test_compose_cli.py`

**Interfaces:**
- Consumes: `_write_png` helper (copy into each test file that needs it), `PanelData(kind="image")`, `SizedCell.kind == "image"`.
- Produces: `draw_panel` draws one `AxesImage` with the axis off and returns `None` for `kind == "image"`; `panel_group_hint` returns `None` for `"image"`.

- [ ] **Step 1: Write the failing tests** —

`tests/test_compose_adapters.py`:

```python
def test_draw_panel_image_axis_off_no_title(tmp_path):
    png = _write_png(tmp_path / "ref.png")
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    p = PanelDef("i", PanelSource(png, "image", {}), show_title=True)
    assert draw_panel(ax, p, load_panel(p), None) is None
    assert len(ax.images) == 1 and not ax.axison and ax.get_title() == ""
```

`tests/test_compose_render.py` (add `_write_png` — same body as Task 5 — near `_write_obl`):

```python
def test_image_panel_lettered_between_maps_sized_by_width_no_colorbar(tmp_path):
    from dfxm.common.plotting import measured_box_in

    h5 = _write_obl(tmp_path / "obl.h5")
    png = _write_png(tmp_path / "ref.png")
    base = render_recipe(_two_panel_recipe(h5, colorbar=True))
    r = _two_panel_recipe(h5, colorbar=True)
    r.panels.append(PanelDef("i", PanelSource(png, "image", {}), width_cm=3.0))
    r.layout.children.insert(1, PanelRef("i"))
    res = render_recipe(r)
    assert res.n_panels == 3 and res.n_rendered == 3
    ax_i = res.axes_by_id["i"]
    assert [t.get_text() for t in ax_i.texts] == ["B"]  # lettered in reading order
    assert [t.get_text() for t in res.axes_by_id["b"].texts] == ["C"]
    w, h = measured_box_in(res.figure, ax_i)
    assert abs(w - 3.0 / 2.54) < 0.005 * w and abs(h - 1.5 / 2.54) < 0.005 * h
    # exactly one extra axes (the image itself): no colourbar axes for it
    assert len(res.figure.axes) == len(base.figure.axes) + 1


def test_image_panel_missing_file_is_placeholder_with_note(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.panels.append(PanelDef("i", PanelSource(str(tmp_path / "gone.png"), "image", {})))
    r.layout.children.append(PanelRef("i"))
    res = render_recipe(r)
    assert res.n_rendered == 2 and any("gone.png" in n and "placeholder" in n for n in res.notes)


def test_united_colorbar_ignores_image_panel_between_maps(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    png = _write_png(tmp_path / "ref.png")
    base = render_recipe(_united_recipe(h5))
    r = _united_recipe(h5)
    r.panels.append(PanelDef("i", PanelSource(png, "image", {})))
    r.layout.children.insert(1, PanelRef("i"))
    res = render_recipe(r)
    assert res.n_rendered == base.n_rendered + 1
    assert len(res.figure.axes) == len(base.figure.axes) + 1
```

(`_united_recipe(h5, *, pos="right")` already exists at `tests/test_compose_render.py:571`; if its layout root is not a flat `Row`, insert the `PanelRef("i")` into whichever container holds the two map refs instead — read the helper first.)

`tests/test_compose_gridmap.py`, inside `test_panel_group_hint_covers_kinds` append:

```python
    assert panel_group_hint(p("image", {})) is None  # neutral grey chip
```

`tests/test_compose_cli.py` (add `_write_png` — same body as Task 5):

```python
def test_cli_renders_recipe_with_image_panel(tmp_path, capsys):
    h5 = _write_obl(tmp_path / "obl.h5")
    png = _write_png(tmp_path / "ref.png")
    r = _two_panel_recipe(h5)
    r.panels.append(PanelDef("i", PanelSource(png, "image", {}), width_cm=2.0))
    r.layout.children.append(PanelRef("i"))
    rp = tmp_path / "r.json"
    rp.write_text(recipe_to_json(r, base_dir=str(tmp_path)))  # image path stored relative
    out = tmp_path / "out"
    assert _main(["render", str(rp), "-o", str(out), "--formats", "png"]) == 0
    assert os.path.exists(out / "demo.png")
```

- [ ] **Step 2: Run them, expect failure**

Run: `python3 -m pytest tests/test_compose_adapters.py tests/test_compose_render.py tests/test_compose_gridmap.py tests/test_compose_cli.py -q -k image`
Expected: FAIL — `StageUserError: unknown panel kind 'image'` from `draw_panel`; the gridmap assertion fails with a string hint.

- [ ] **Step 3: Implement** —

`adapters.draw_panel`, insert right after the placeholder branch (`return None` at line ≈349) and before `from ..common.plotting import resolve_cmap`:

```python
    if data.kind == "image":
        img = data.payload["image"]
        # the cell already has the image's aspect, so "auto" fills it edge to
        # edge; no title, colourbar or scale bar for an external image
        ax.imshow(img, interpolation="none", aspect="auto", cmap="gray" if img.ndim == 2 else None)
        ax.set_axis_off()
        return None
```

`render.py` draw loop: change `else:  # placeholder` to `else:  # image or placeholder — no colourbar, no scale bar` (the call already matches what an image needs; keep it one branch).

`gridmap.panel_group_hint`, before `key = "volume_id" if ...`:

```python
    if src.kind == "image":
        return None  # an external image belongs to no quantity group: neutral chip
```

- [ ] **Step 4: Run the four test files**

Run: `python3 -m pytest tests/test_compose_adapters.py tests/test_compose_render.py tests/test_compose_gridmap.py tests/test_compose_cli.py -q`
Expected: all PASS. If `test_image_panel_lettered…` fails on the axes count, inspect `res.figure.axes` — a per-panel scale bar is an offsetbox, not an axes, so the only legitimate extra axes are the two map colourbars and the image; fix the renderer, not the assertion.

- [ ] **Step 5: Docs** — `docs/Codebase.md`:
  - `draw_panel` bullet: add "`image` (2026-09-02): `ax.imshow(payload["image"], interpolation="none", aspect="auto")` (grey colormap for a 2-D file), `ax.set_axis_off()`, returns `None`; `show_title`/`titled`, `clim`, `cmap`, `colorbar` and `scale_bar` are ignored — an external image has none of those."
  - `panel_group_hint` bullet: add "`"image"` → `None` (neutral chip)".
  - render draw-loop paragraph: where the three branches (`"map"`/`"trace"`/placeholder) are described, say the else branch also serves `"image"` cells, which get no colourbar axes and are excluded from the `("map_layer", "slice_plane", "profiles_ref")` scale-bar tuples and (via `group=None`) from united colourbars.

- [ ] **Step 6: Full suite + commit**

Run: `python3 -m pytest -q --ignore=tests/test_gui_viewer3d.py && ruff check .`
Expected: all PASS, no ruff findings.

```bash
git add dfxm/compose/adapters.py dfxm/compose/render.py dfxm/compose/gridmap.py tests/test_compose_adapters.py tests/test_compose_render.py tests/test_compose_gridmap.py tests/test_compose_cli.py docs/Codebase.md
git commit -m "feat(compose): render 'image' panels — lettered, axis-off, no colourbar"
```

---

### Task 8: GUI — Add image…, Width (cm), per-kind inspector enable rule, Usage docs

**Files:**
- Modify: `gui/figure_builder.py:256-258` (button), `:932-1022` (`_build_override_editor`), `:1037-1074` (`_load_panel_page`), `:1102-1169` (the three `"profiles_trace"` checks), `:1189-1198` (getters), `:1270-1287` (`changes`), `:1416-1426` (new `add_image` next to `add_spacer`), `:1712` (new `_on_add_image` next to `_on_add_panels`), module constants near `_TRI_STATE` (line 93)
- Modify: `docs/Usage.md:2220-2224` (Panels concept), `:2251` (new Add image… paragraph after the Add panels… steps), `:2494-2525` (Panel page), `:2545-2555` (Spacer bullet)
- Modify: `docs/Codebase.md:1925-1931` (`figure_builder.py` row)
- Test: `tests/test_gui_figure_builder.py`

**Interfaces:**
- Consumes: `PanelDef.width_cm`, `PANEL_KINDS` incl. `"image"`, `add_panels(panels) -> dict[str, str]`.
- Produces: `FigureBuilderWindow.add_image(path: str) -> dict[str, str]`; `_on_add_image()` (file dialog slot); `_ov_width: QDoubleSpinBox`; module constant `_IMAGE_KINDS = ("map_layer", "slice_plane", "profiles_ref")`; override key `"width_cm"`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_gui_figure_builder.py` (add a `_write_png` helper with the Task 5 body; `numpy` imported inside it):

```python
def _write_png(path, w=40, h=20):
    import numpy as np
    from matplotlib.image import imsave

    rgb = np.zeros((h, w, 3), "f4")
    rgb[..., 0] = np.linspace(0.0, 1.0, w)[None, :]
    imsave(str(path), rgb)
    return str(path)


def test_add_image_appends_panel_and_inspector_enables_only_width(tmp_path):
    png = _write_png(tmp_path / "ref.png")
    w = _win()
    w.add_panels([_panel("a")])
    w.add_image(png)
    r = w.recipe()
    img = r.panels[-1]
    assert img.source.kind == "image" and img.source.h5_path == png and img.title == "ref.png"
    assert img.id == "image_1"
    leaf = r.layout.children[-1]
    assert isinstance(leaf, PanelRef) and leaf.panel_id == img.id
    w._select_outline_panel(img.id)
    assert w._ov_width.isEnabled() and w._ov_roi.isEnabled()
    for widget in (
        w._ov_clim,
        w._ov_cmap,
        w._ov_colorbar,
        w._ov_scale,
        w._ov_crop,
        w._ov_roi_pick,
        w._ov_roi_all,
        w._ov_show_title,
        w._ov_ynums,
    ):
        assert not widget.isEnabled()
    assert w._ov_label_mode.isEnabled()
    w._ov_width.setValue(4.0)  # real widget signal path
    assert img.width_cm == 4.0
    w._ov_width.setValue(0.0)  # special value = default
    assert img.width_cm is None
    w.add_image(png)
    assert w.recipe().panels[-1].id == "image_2"
    w._select_outline_panel("a")
    assert not w._ov_width.isEnabled() and w._ov_clim.isEnabled() and w._ov_roi_pick.isEnabled()


def test_trace_panel_inspector_keeps_scale_and_title_enabled():
    w = _win()
    w.add_panels([PanelDef("t", PanelSource("/x.h5", "profiles_trace", {"job": {}, "field": "s"}))])
    w._select_outline_panel("t")
    assert w._ov_scale.isEnabled() and w._ov_show_title.isEnabled() and w._ov_ynums.isEnabled()
    for widget in (w._ov_roi, w._ov_roi_pick, w._ov_roi_all, w._ov_crop, w._ov_clim, w._ov_cmap, w._ov_colorbar, w._ov_width):
        assert not widget.isEnabled()


def test_copy_roi_to_all_skips_image_panels(tmp_path):
    png = _write_png(tmp_path / "ref.png")
    w = _win()
    w.add_panels([_panel("a"), _panel("b")])
    w.add_image(png)
    r = w.recipe()
    r.panels[0].roi = (0, 2, 0, 3)
    w._select_outline_panel("a")
    w._on_copy_roi_to_all()
    assert r.panels[1].roi == (0, 2, 0, 3) and r.panels[2].roi is None


def test_image_panel_renders_in_builder_preview(tmp_path):
    png = _write_png(tmp_path / "ref.png")
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    w.add_image(png)
    res = render_and_wait(w)
    assert res is not None and res.n_rendered == 2
```

- [ ] **Step 2: Run them, expect failure**

Run: `python3 -m pytest tests/test_gui_figure_builder.py -q -k "add_image or trace_panel_inspector or skips_image or image_panel_renders"`
Expected: FAIL — `AttributeError: ... has no attribute 'add_image'`.

- [ ] **Step 3: Module constant + button + slot + mutator** —

Next to `_TRI_STATE` (line ≈93):

```python
# Panel kinds that are a picture of the sample: they have an ROI picker,
# colour limits, a colormap, a colourbar and a µm/cm scale. Traces and
# external images are not.
_IMAGE_KINDS = ("map_layer", "slice_plane", "profiles_ref")
```

After the `add_btn` block (line ≈258):

```python
        add_img_btn = QPushButton("Add image…")
        add_img_btn.setToolTip(
            "Place a PNG/JPEG/TIFF — e.g. a panel reproduced from another paper — as a "
            "lettered panel. Set its printed width in the inspector; the height follows "
            "the image's own proportions."
        )
        add_img_btn.clicked.connect(self._on_add_image)
        layout.addWidget(add_img_btn)
```

After `add_scale_bar` (line ≈1426):

```python
    def add_image(self, path: str) -> dict[str, str]:
        """Append *path* (a PNG/JPEG/TIFF) as a lettered ``"image"`` panel, titled
        with the file name; ids run ``image_1``, ``image_2``, … (``add_panels``
        still uniquifies on a collision). Returns ``add_panels``'s rename map."""
        n = 1 + sum(1 for p in self._recipe.panels if p.source.kind == "image")
        panel = PanelDef(
            f"image_{n}", PanelSource(path, "image", {}), title=os.path.basename(path)
        )
        return self.add_panels([panel])
```

After `_on_add_panels` (line ≈1718):

```python
    def _on_add_image(self) -> None:
        start = os.path.dirname(self._current_path) if self._current_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add image panel",
            start,
            "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All files (*)",
        )
        if path:
            self.add_image(path)
```

- [ ] **Step 4: Width spin + wiring** —

In `_build_override_editor`, after the `form.addRow("Panel scale", self._ov_scale)` line:

```python
        self._ov_width = QDoubleSpinBox()
        self._ov_width.setRange(0.0, 30.0)
        self._ov_width.setDecimals(2)
        self._ov_width.setSuffix(" cm")
        self._ov_width.setSpecialValueText("default (6 cm)")
        self._ov_width.setToolTip(
            "Printed width of an image panel; its height follows the image's own "
            "proportions. 0 = the default 6 cm. Image panels only."
        )
        self._ov_width.valueChanged.connect(lambda _v: self._on_override_field_edited("width_cm"))
        form.addRow("Width", self._ov_width)
```

`_on_override_field_edited` getters: add `"width_cm": self._ov_width.value,`.

`_apply_panel_overrides`, after the `scale_um_per_cm` block:

```python
        if "width_cm" in values:
            width = values["width_cm"]
            changes["width_cm"] = float(width) if width else None
```

`_load_panel_page`: add `self._ov_width,` to the `widgets` tuple; after `self._ov_scale.setValue(panel.scale_um_per_cm or 0.0)` add `self._ov_width.setValue(panel.width_cm or 0.0)`.

- [ ] **Step 5: Per-kind enable rule** — in `_load_panel_page`, delete the Task 3 line `self._ov_ynums.setEnabled(panel.source.kind == "profiles_trace")` and, after the closing `for w in widgets: w.blockSignals(False)` loop, add:

```python
        # Which overrides mean anything for this kind (see docs/Usage.md, Panel page)
        kind = panel.source.kind
        is_trace, is_image = kind == "profiles_trace", kind == "image"
        is_map = kind in _IMAGE_KINDS
        self._ov_roi.setEnabled(not is_trace)  # images: a plain pixel crop
        for w in (self._ov_roi_pick, self._ov_roi_all, self._ov_crop, self._ov_clim, self._ov_cmap, self._ov_colorbar):
            w.setEnabled(is_map)
        self._ov_scale.setEnabled(not is_image)  # traces use it as their trace scale
        self._ov_show_title.setEnabled(not is_image)
        self._ov_width.setEnabled(is_image)
        self._ov_ynums.setEnabled(is_trace)
```

Replace the three `"profiles_trace"` kind checks:

- `_on_pick_panel_roi` (≈1110): `if panel.source.kind not in _IMAGE_KINDS:` with notes text `"only map, slice and reference panels have an ROI to pick"`.
- `_on_pick_panel_roi` preview list (≈1124): `p.source.kind in _IMAGE_KINDS`.
- `_on_copy_roi_to_all` (≈1160): `if p is src or p.source.kind not in _IMAGE_KINDS:`.

- [ ] **Step 6: Run the builder tests**

Run: `python3 -m pytest tests/test_gui_figure_builder.py -q`
Expected: all PASS (including the Task 3 test, whose trace-only assertion still holds under the new rule).

- [ ] **Step 7: Usage docs** — `docs/Usage.md`:

  - **Concepts → Panels** (≈2220): after "…or a `profiles` job's reference image/line trace)" add ", or — since 2026-09-02 — an external **image file** (PNG/JPEG/TIFF, see **Add image…** below)".
  - After the **Add panels…** two-step list and before **Arrange…** (≈2298), add:

```
**Add image…**

Places a picture file — typically a panel reproduced from another paper, or a
schematic drawn elsewhere — as an ordinary panel. Pick a PNG/JPEG/TIFF; it
joins the current container titled with its file name, takes the next letter
in reading order like any panel, and is sized by its printed **Width** (cm,
inspector; default 6 cm) with the height following the picture's own
proportions. A pinned row height or column width overrides that width the
same way it does for a map. It never gets a title, colourbar or scale bar,
and it takes no part in united colourbars or trace autoscaling. The **ROI
crop** box still works as a plain pixel crop (type `r0,r1,c0,c1`; the
picker is not offered). The recipe stores the image path relative to the
`.json`, so keep the picture beside the recipe; a missing file renders as a
placeholder rather than failing the figure.
```

  - **Panel** bullet (≈2494): after the "trace panels have no ROI" parenthesis note that image panels accept the text box only; after the panel-scale clause add "and, for image panels, **Width** in cm (0 = the 6 cm default). Controls that mean nothing for the selected kind are greyed out: maps get everything except Width and Y-axis numbers; traces get ROI-less Label, Show title, Panel scale (their trace scale) and Y-axis numbers; images get ROI text, Label and Width."
  - **Spacer** bullet (≈2545): replace the sentence from "Besides plain breathing room…" through "…LaTeX/slide layer." with: "Besides plain breathing room, a spacer reserves an exact-size empty spot — for content you will overlay in a vector editor. A picture you already have as a file no longer needs this: **Add image…** places it as a real, lettered panel." Keep the two caveats that follow.

- [ ] **Step 8: Codebase docs** — `docs/Codebase.md` `figure_builder.py` row: after the **Add panels…** clause add "an **Add image…** button (2026-09-02, `_on_add_image()` → `QFileDialog.getOpenFileName` with an image filter, starting in the current recipe's directory, then `add_image(path)`); `add_image(path) -> dict[str, str]` builds `PanelDef(f"image_{n}", PanelSource(path, "image", {}), title=basename)` — `n` counting existing image panels — and hands it to `add_panels`". In the **Panel page** sentence add "`self._ov_width` (`QDoubleSpinBox`, cm, special value `0` = default → `width_cm=None`)", add `"width_cm"` to the fixed field-name list, and append: "`_load_panel_page` ends with the per-kind enable rule over module constant `_IMAGE_KINDS = ("map_layer", "slice_plane", "profiles_ref")`: ROI text off for traces; Pick…/→ all maps/Crop/clim/cmap/Colourbar only for `_IMAGE_KINDS`; Panel scale and Show title off for images; Width only for images; Y-axis numbers only for traces. `_on_pick_panel_roi` and `_on_copy_roi_to_all` test `kind in _IMAGE_KINDS` rather than `!= "profiles_trace"` so image panels are skipped too."

- [ ] **Step 9: Full suite + lint + commit**

Run: `python3 -m pytest -q --ignore=tests/test_gui_viewer3d.py && ruff check .`
Expected: all PASS, no ruff findings.

```bash
git add gui/figure_builder.py tests/test_gui_figure_builder.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): Add image… panels with printed width; per-kind inspector controls"
```

---

### Task 9: Eyeball in the real GUI (no code)

- [ ] **Step 1:** Launch `python3 -m gui.app` from a directory that is not a data directory, open Figure builder, add one map or slice panel from an existing recipe, then **Add image…** with any PNG. Confirm: the image appears lettered, the Width spin resizes it live, a missing file (rename it, Refresh data) shows the hatched placeholder.
- [ ] **Step 2:** Add a profiles trace panel, untick **Y-axis numbers**, confirm the numbers and offset go while the y-label stays and the panel's left margin tightens.
- [ ] **Step 3:** Save the recipe next to the PNG, reopen it, confirm the image path resolved.
- [ ] **Step 4:** Record the outcome in the handoff memory (`/handoff-note`).
