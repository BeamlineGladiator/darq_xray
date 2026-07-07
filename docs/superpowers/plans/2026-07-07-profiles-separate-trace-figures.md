# Profiles Separate Trace Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export each profiled field of the `profiles` stage as its own full-size line-profile figure with a user-settable aspect ratio, line thickness, curve colour, and font scale, keeping the existing stacked "companion" figure behind a toggle.

**Architecture:** Add trace-figure rendering primitives to `dfxm/stages/profiles.py`, five new appearance/output params plus a `save_companion` toggle to the stage spec, wire them through `run()` and the publication-export `figures()` catalog, and update the GUI results summary/preview. The companion path (`build_companion_figure`) and its pinned regression look are left untouched.

**Tech Stack:** Python, NumPy, h5py, matplotlib (explicit `Figure` API), pytest. GUI is PySide6 but only `gui/stage_view.py` module-level helpers are touched (no Qt objects).

## Global Constraints

- Keep `dfxm/` Qt-free — no PySide6/pyvista imports in `dfxm/stages/profiles.py`.
- Build figures with the explicit `matplotlib.figure.Figure` API via `styled_figure(...)`; never `pyplot`/`matplotlib.use(...)`.
- Ruff config: line length 100, double quotes, target py310, rules E/F/I. `ruff format` runs automatically on Write/Edit via the settings hook.
- Input-validation failures raise `StageUserError(message, hint=...)` from `dfxm.common.errors`.
- Every new `Param` needs a first-time-user `help`; advanced params need a `group` (`tests/test_param_metadata.py` enforces this).
- **Docs contract:** update `docs/Usage.md` (user-facing) and `docs/Codebase.md` (code reference) in the SAME task/commit as the code change they describe.
- Do NOT modify `build_companion_figure`, `save_companion_figure`, or `render_single` — the companion look is pinned by `tests/test_figures_catalog.py` and `tests/test_export_fidelity.py`.
- New trace figures are `kind="plot"` (no scale bar), consistent with the companion.
- Trace text is scaled by the trace figures' own `trace_font_scale`, NOT the map `style.font_scale`. A blank `trace_color` falls back to `"C0"`.

---

### Task 1: Trace-figure rendering primitives

**Files:**
- Modify: `dfxm/stages/profiles.py` (add `parse_aspect`, `build_trace_figure`, `_save_traces`; import already has `StageUserError`, `styled_figure`, `os`, `np`, `Figure`)
- Test: `tests/test_stage_profiles.py`
- Docs: `docs/Codebase.md:362` region (profiles function list)

**Interfaces:**
- Produces:
  - `parse_aspect(s: str) -> tuple[float, float]` — parse `"W:H"` into positive `(w, h)` floats; raises `StageUserError` on bad input.
  - `build_trace_figure(fld, geom, *, aspect_wh, width_in, linewidth, color, font_scale, style=None) -> Figure` — one standalone trace figure for a single field; no `savefig`. `fld` is a `_collect` field dict (`{"vid", "attrs", "value_mean", "value_std", ...}`); `geom` is a `line_geometry` dict (needs `geom["distance"]`, `geom["L"]`). `aspect_wh` is `(w, h)`.
  - `_save_traces(out_dir, stem, fields, geom, *, aspect, width_in, linewidth, color, font_scale, dpi, style=None) -> list[str]` — build+save one `{stem}__trace__{vid}.png` per field; returns the paths.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stage_profiles.py` (after the existing imports; `pytest` and `StageUserError` imports may need adding at top — add `import pytest` and `from dfxm.common.errors import StageUserError` if absent):

```python
def _fake_field(vid="strain", *, std=True):
    n = 20
    dist = np.linspace(0.0, 10.0, n)
    vm = np.sin(dist)
    vs = np.full(n, 0.1) if std else None
    fld = {
        "vid": vid,
        "attrs": {"cbar_label": "c", "kind": vid, "source_volume": "", "title": vid, "cmap": "gray"},
        "value_mean": vm,
        "value_std": vs,
    }
    geom = {"distance": dist, "L": 10.0}
    return fld, geom


def test_parse_aspect_valid():
    assert PR.parse_aspect("4:3") == (4.0, 3.0)
    assert PR.parse_aspect("1:1") == (1.0, 1.0)
    assert PR.parse_aspect("1.5:1") == (1.5, 1.0)


@pytest.mark.parametrize("bad", ["", "4", "4:0", "a:b", "1:2:3", "-4:3"])
def test_parse_aspect_invalid_raises(bad):
    with pytest.raises(StageUserError):
        PR.parse_aspect(bad)


def test_build_trace_figure_aspect_linewidth_color():
    from matplotlib.colors import to_rgba

    fld, geom = _fake_field(std=True)
    fig = PR.build_trace_figure(
        fld, geom, aspect_wh=(2.0, 1.0), width_in=6.0, linewidth=3.5, color="red", font_scale=1.0
    )
    w, h = fig.get_size_inches()
    assert abs(w - 6.0) < 1e-6 and abs(h - 3.0) < 1e-6
    line = fig.axes[0].lines[0]
    assert abs(line.get_linewidth() - 3.5) < 1e-9
    assert to_rgba(line.get_color()) == to_rgba("red")


def test_build_trace_figure_blank_color_defaults_c0():
    from matplotlib.colors import to_rgba

    fld, geom = _fake_field(std=True)
    fig = PR.build_trace_figure(
        fld, geom, aspect_wh=(4.0, 3.0), width_in=6.0, linewidth=2.0, color="", font_scale=1.0
    )
    line = fig.axes[0].lines[0]
    assert to_rgba(line.get_color()) == to_rgba("C0")


def test_build_trace_figure_no_std_no_band():
    fld, geom = _fake_field(std=False)
    fig = PR.build_trace_figure(
        fld, geom, aspect_wh=(4.0, 3.0), width_in=6.0, linewidth=2.0, color="", font_scale=1.0
    )
    assert len(fig.axes[0].collections) == 0  # no fill_between std band
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_profiles.py -q -k "parse_aspect or build_trace_figure"`
Expected: FAIL — `AttributeError: module 'dfxm.stages.profiles' has no attribute 'parse_aspect'`.

- [ ] **Step 3: Implement `parse_aspect`**

Insert into `dfxm/stages/profiles.py` in the "Style helpers" section (just above `auto_line_color`, near line 368):

```python
def parse_aspect(s: str) -> tuple[float, float]:
    """Parse a 'W:H' aspect string into positive (width, height) floats."""
    parts = str(s).split(":")
    hint = "Enter the ratio as two numbers separated by a colon, e.g. 4:3 or 1:1."
    if len(parts) != 2:
        raise StageUserError(f"aspect must be 'W:H' (got {s!r})", hint=hint)
    try:
        w, h = float(parts[0]), float(parts[1])
    except ValueError:
        raise StageUserError(f"aspect must be 'W:H' with numeric parts (got {s!r})", hint=hint) from None
    if not (np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0):
        raise StageUserError(f"aspect parts must be positive and finite (got {s!r})", hint=hint)
    return w, h
```

- [ ] **Step 4: Implement `build_trace_figure` and `_save_traces`**

Insert `build_trace_figure` just after `build_companion_figure` / `save_companion_figure` (near line 517, before `render_single`):

```python
def build_trace_figure(
    fld,
    geom,
    *,
    aspect_wh,
    width_in,
    linewidth,
    color,
    font_scale,
    style: PlotStyle | None = None,
) -> Figure:
    """Build a standalone line-profile figure for a single field. Does NOT savefig.

    Figure size is ``(width_in, width_in * h / w)`` for ``aspect_wh == (w, h)``.
    All trace text is multiplied by ``font_scale`` — this is the trace figures'
    own scale, independent of the map figures' ``style.font_scale``. The curve
    and its std band use ``color`` (blank/None -> ``"C0"``). No colorbar (it is a
    1-D plot); ``style`` only selects the ``styled_figure`` layout engine so the
    background/layout matches the rest of the stage's output.
    """
    w_ratio, h_ratio = aspect_wh
    fs = float(font_scale)
    curve_color = color or "C0"
    fig = styled_figure(
        (float(width_in), float(width_in) * float(h_ratio) / float(w_ratio)),
        styled=style is not None,
    )
    ax = fig.add_subplot(111)
    distance = geom["distance"]
    vm = fld["value_mean"]
    ax.plot(distance, vm, "-", lw=float(linewidth), color=curve_color, zorder=3)
    if fld["value_std"] is not None:
        vs = fld["value_std"]
        ax.fill_between(distance, vm - vs, vm + vs, color=curve_color, alpha=0.22, lw=0, zorder=2)
    ax.set_ylabel(fld["attrs"]["cbar_label"], fontsize=10 * fs)
    src = os.path.basename(fld["attrs"]["source_volume"]) or "(consolidated)"
    ax.set_title(f"{fld['attrs']['kind']}  |  {fld['vid']}  |  {src}", fontsize=10 * fs, loc="left")
    ax.grid(True, color="0.85", lw=0.6)
    ax.set_xlim(0.0, geom["L"])
    ax.set_xlabel("distance along line (µm)", fontsize=12 * fs)
    ax.tick_params(axis="both", labelsize=10 * fs)
    return fig
```

Insert `_save_traces` in the Drivers section, just after `_save_overviews` (near line 631):

```python
def _save_traces(
    out_dir, stem, fields, geom, *, aspect, width_in, linewidth, color, font_scale, dpi, style=None
):
    aspect_wh = parse_aspect(aspect)
    paths = []
    for fld in fields:
        tr_png = os.path.join(out_dir, f"{stem}__trace__{fld['vid']}.png")
        fig = build_trace_figure(
            fld,
            geom,
            aspect_wh=aspect_wh,
            width_in=width_in,
            linewidth=linewidth,
            color=color,
            font_scale=font_scale,
            style=style,
        )
        fig.savefig(tr_png, dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight")
        paths.append(tr_png)
    return paths
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_profiles.py -q -k "parse_aspect or build_trace_figure"`
Expected: PASS (7 tests).

- [ ] **Step 6: Update `docs/Codebase.md`**

In the `profiles.py` function list (near line 362, after the `save_companion_figure` bullet), add:

```markdown
- `parse_aspect(s)` — parse a `"W:H"` aspect string into positive `(w, h)` floats; raises `StageUserError` (with hint) on anything that isn't two positive finite numbers.
- `build_trace_figure(fld, geom, *, aspect_wh, width_in, linewidth, color, font_scale, style=None)` — build and return a standalone line-profile `Figure` for a single field (distance vs value_mean + optional std band). Size is `(width_in, width_in*h/w)`; all trace text is multiplied by `font_scale` (its own scale, independent of the map `style.font_scale`); the curve/band use `color` (blank/`None` → `"C0"`). `kind="plot"`, no colorbar. Does NOT call `savefig`.
- `_save_traces(out_dir, stem, fields, geom, *, aspect, width_in, linewidth, color, font_scale, dpi, style=None)` — build+save one `{stem}__trace__{vid}.png` per field; returns the paths.
```

- [ ] **Step 7: Run ruff and commit**

Run: `ruff check dfxm/stages/profiles.py tests/test_stage_profiles.py`
Expected: no errors.

```bash
git add dfxm/stages/profiles.py tests/test_stage_profiles.py docs/Codebase.md
git commit -m "feat(profiles): trace-figure rendering primitives (parse_aspect, build_trace_figure, _save_traces)"
```

---

### Task 2: Stage params, `run()` wiring, and `traces` result field

**Files:**
- Modify: `dfxm/stages/profiles.py` (`STAGE.params`, `ProfileJobResult`, `run()`)
- Test: `tests/test_stage_profiles.py`
- Docs: `docs/Usage.md` (section 8), `docs/Codebase.md` (profiles result + run notes)

**Interfaces:**
- Consumes: `parse_aspect`, `build_trace_figure`, `_save_traces` (Task 1).
- Produces:
  - `ProfileJobResult.traces: list[str]` (default `[]`).
  - New stage params: `save_traces` (BOOL, default `True`), `save_companion` (BOOL, default `True`), `trace_aspect` (STR, default `"4:3"`), `trace_width_in` (FLOAT, default `6.0`), `trace_linewidth` (FLOAT, default `2.0`), `trace_color` (STR, default `""`), `trace_font_scale` (FLOAT, default `1.4`).
  - `run()` writes the companion only when `save_companion` (else `jr.figure is None`) and writes traces into `jr.traces` when `save_traces`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stage_profiles.py`:

```python
def _base_params(h5, out, **extra):
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1,"fig_name":"prof0"}]'
    )
    return {
        "consolidated_h5": str(h5),
        "mode": "parameter",
        "jobs_json": jobs,
        "output_dir": str(out),
        **extra,
    }


def test_run_writes_traces_by_default(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    res = PR.run(_base_params(h5, out))
    jr = res.jobs[0]
    assert jr.figure and os.path.exists(jr.figure)  # companion on by default
    assert len(jr.traces) == 2 and all(os.path.exists(t) for t in jr.traces)
    assert all("__trace__" in t for t in jr.traces)


def test_run_companion_off_yields_no_companion(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    res = PR.run(_base_params(h5, out, save_companion=False))
    jr = res.jobs[0]
    assert jr.figure is None
    assert not os.path.exists(os.path.join(str(out), "prof0.png"))
    assert len(jr.traces) == 2


def test_run_traces_off_keeps_companion(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    res = PR.run(_base_params(h5, out, save_traces=False))
    jr = res.jobs[0]
    assert jr.figure and os.path.exists(jr.figure)
    assert jr.traces == []


def test_run_bad_aspect_raises(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    with pytest.raises(StageUserError):
        PR.run(_base_params(h5, out, trace_aspect="oops"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_profiles.py -q -k "traces or companion_off or bad_aspect"`
Expected: FAIL — `AttributeError: 'ProfileJobResult' object has no attribute 'traces'` (and/or unknown param behaviour).

- [ ] **Step 3: Add the `traces` field to `ProfileJobResult`**

In `dfxm/stages/profiles.py`, in `ProfileJobResult` (near line 217), add after `overviews`:

```python
    traces: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Add the new stage params**

In `STAGE.params`, insert this block immediately before the `output_dir` param (near line 207):

```python
        Param(
            "save_traces",
            ParamType.BOOL,
            "Save traces",
            default=True,
            advanced=True,
            group="Output",
            help="Write each profiled field as its own line-profile figure (separate from the stacked companion).",
        ),
        Param(
            "save_companion",
            ParamType.BOOL,
            "Save companion",
            default=True,
            advanced=True,
            group="Output",
            help=(
                "Also write the stacked companion figure (overview image + all traces in one). "
                "Turn off to export only the separate traces."
            ),
        ),
        Param(
            "trace_aspect",
            ParamType.STR,
            "Trace aspect",
            default="4:3",
            advanced=True,
            group="Appearance",
            help="Aspect ratio (width:height) of each separate trace figure, e.g. 4:3, 1:1, 16:9.",
        ),
        Param(
            "trace_width_in",
            ParamType.FLOAT,
            "Trace width",
            unit="in",
            default=6.0,
            advanced=True,
            group="Appearance",
            help="Width of each separate trace figure in inches; the height follows the aspect ratio.",
        ),
        Param(
            "trace_linewidth",
            ParamType.FLOAT,
            "Trace line width",
            unit="pt",
            default=2.0,
            advanced=True,
            group="Appearance",
            help="Line thickness of the plotted profile curve on the separate trace figures.",
        ),
        Param(
            "trace_color",
            ParamType.STR,
            "Trace colour",
            default="",
            advanced=True,
            group="Appearance",
            help=(
                "Colour of the profile curve and its std band on the separate trace figures "
                "(blank = default matplotlib blue)."
            ),
        ),
        Param(
            "trace_font_scale",
            ParamType.FLOAT,
            "Trace font scale",
            default=1.4,
            advanced=True,
            group="Appearance",
            help=(
                "Multiplies the label/tick/title font size of the separate trace figures "
                "(independent of the map figures' font scale)."
            ),
        ),
```

- [ ] **Step 5: Wire `run()`**

In `run()`, after `dpi = int(p["fig_dpi"])` (near line 666), add:

```python
    save_traces = bool(p["save_traces"])
    save_companion = bool(p["save_companion"])
    trace_aspect = p["trace_aspect"]
    trace_width_in = float(p["trace_width_in"])
    trace_linewidth = float(p["trace_linewidth"])
    trace_color = p["trace_color"] or None
    trace_font_scale = float(p["trace_font_scale"])
    if save_traces:
        parse_aspect(trace_aspect)  # fail fast on a bad aspect before the h5 loop
```

Then replace the parameter-mode write block (currently near lines 710-728):

```python
            color = auto_line_color(ref[3]["cmap"], line_override)
            stem = job.get("fig_name") or f"profile_{name}_{off_used:+.2f}um".replace(
                "+", "p"
            ).replace("-", "m")
            out_png = os.path.join(out_dir, f"{stem}.png")
            save_companion_figure(ref, fields, geom, color, out_png, dpi, style=style)
            jr = ProfileJobResult(
                name=name,
                offset_used_um=off_used,
                figure=out_png,
                fields=[fl["vid"] for fl in fields],
            )
            if bool(p["save_csv"]):
                jr.csvs = _write_csvs(out_dir, stem, geom["distance"], fields)
            if bool(p["save_overview"]):
                jr.overviews = _save_overviews(
                    out_dir, stem, ref, fields, geom, off_used, line_override, dpi, style=style
                )
            result.jobs.append(jr)
```

with:

```python
            color = auto_line_color(ref[3]["cmap"], line_override)
            stem = job.get("fig_name") or f"profile_{name}_{off_used:+.2f}um".replace(
                "+", "p"
            ).replace("-", "m")
            jr = ProfileJobResult(
                name=name,
                offset_used_um=off_used,
                fields=[fl["vid"] for fl in fields],
            )
            if save_companion:
                out_png = os.path.join(out_dir, f"{stem}.png")
                save_companion_figure(ref, fields, geom, color, out_png, dpi, style=style)
                jr.figure = out_png
            if save_traces:
                jr.traces = _save_traces(
                    out_dir,
                    stem,
                    fields,
                    geom,
                    aspect=trace_aspect,
                    width_in=trace_width_in,
                    linewidth=trace_linewidth,
                    color=trace_color,
                    font_scale=trace_font_scale,
                    dpi=dpi,
                    style=style,
                )
            if bool(p["save_csv"]):
                jr.csvs = _write_csvs(out_dir, stem, geom["distance"], fields)
            if bool(p["save_overview"]):
                jr.overviews = _save_overviews(
                    out_dir, stem, ref, fields, geom, off_used, line_override, dpi, style=style
                )
            result.jobs.append(jr)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_profiles.py tests/test_param_metadata.py -q`
Expected: PASS (new run tests green; param-metadata still green with the seven new params).

- [ ] **Step 7: Update `docs/Usage.md` (section 8)**

Change the **Output** line (line 513) from:

```markdown
- **Output:** a stacked companion figure + per-field CSVs + per-field overviews.
```

to:

```markdown
- **Output:** one line-profile figure **per field** (`<fig_name>__trace__<field>.png`) + per-field CSVs + per-field overviews, plus (optionally) the stacked companion figure.
```

Then add a new subsection immediately after the "Jobs JSON" example block (after the closing ```` ``` ```` of the JSON example near line 550-555):

```markdown
#### Separate trace figures (per field)

By default each profiled field is written as its **own** figure
(`<fig_name>__trace__<field>.png`) so a single trace — say misorientation vs
distance — reads clearly as a paper subfigure. Shape and style them with:

| Param | Meaning |
|---|---|
| `save_traces` | Write the separate per-field trace figures (default on). |
| `save_companion` | Also write the old stacked companion figure (overview + all traces in one). Turn off for traces-only. |
| `trace_aspect` | Aspect ratio `width:height` of each trace figure — `4:3`, `1:1`, `16:9`, … |
| `trace_width_in` | Width of each trace figure in inches; the height follows the aspect. |
| `trace_linewidth` | Thickness (pt) of the plotted profile curve. |
| `trace_color` | Colour of the curve and its std band (blank = default matplotlib blue). |
| `trace_font_scale` | Multiplies the trace figures' label/tick/title fonts, independent of the map figures' font scale. |

The overview images (plane + line, per field) are still written by
`save_overview` and are unaffected by these knobs.
```

- [ ] **Step 8: Update `docs/Codebase.md` (profiles result + run)**

Update the `ProfileJobResult` bullet (line 359) from:

```markdown
- `ProfileJobResult` / `ProfilesResult`.
```

to:

```markdown
- `ProfileJobResult` (now carries `traces: list[str]` — the per-field standalone trace PNGs; `figure` is `None` when `save_companion` is off) / `ProfilesResult`.
```

Update the drivers bullet (line 366) so the `run` sentence reads:

```markdown
`_write_csvs`, `_save_overviews`, `_save_traces`; `run` supports `parameter` (per-field trace figures when `save_traces`, the stacked companion when `save_companion`, plus CSVs/overviews) and `preview` modes; a bad `trace_aspect` raises `StageUserError` up front.
```

- [ ] **Step 9: Run ruff and commit**

Run: `ruff check dfxm/stages/profiles.py tests/test_stage_profiles.py`
Expected: no errors.

```bash
git add dfxm/stages/profiles.py tests/test_stage_profiles.py docs/Usage.md docs/Codebase.md
git commit -m "feat(profiles): per-field trace figures + aspect/width/linewidth/color/font-scale params + save_companion toggle"
```

---

### Task 3: Publication export — trace `FigureSpec`s in `figures()`

**Files:**
- Modify: `dfxm/stages/profiles.py` (`figures()`)
- Test: `tests/test_figures_catalog.py` (update 3 existing tests, add 3), `tests/test_export_fidelity.py` (update 1 existing)
- Docs: `docs/Codebase.md` (figures bullet + data-flow row), `docs/Usage.md` (catalog table + data-flow row)

**Interfaces:**
- Consumes: `build_trace_figure`, `parse_aspect`, `_collect`, `ProfileJobResult.fields`/`.traces` (Tasks 1-2).
- Produces: `figures()` emits, per parameter-mode job, a companion `FigureSpec` (`figure_id=f"profiles_{name}"`) when `save_companion`, followed by one trace `FigureSpec` per field (`figure_id=f"profiles_{name}__trace__{vid}"`, `filename=f"{fig_name}__trace__{vid}"`) when `save_traces`. Toggles read from `params` with `STAGE.defaults()` fallback (so both default on).

- [ ] **Step 1: Update the existing catalog tests to the new counts**

In `tests/test_figures_catalog.py`:

`test_profiles_catalog_one_spec_per_job` (near line 1182) — replace the assertions after `specs = Profiles.figures(params)` with:

```python
    specs = Profiles.figures(res, params)
    # 1 companion + 1 trace per field (raw_sum, strain), both toggles default on
    assert len(specs) == 3
    ids = {s.figure_id for s in specs}
    assert "profiles_oblique_full" in ids
    assert "profiles_oblique_full__trace__raw_sum" in ids
    assert "profiles_oblique_full__trace__strain" in ids
    assert all(s.kind == "plot" for s in specs)
```

`test_profiles_catalog_distinct_ids_multi_job` (near line 1199) — `jr_a`/`jr_b` each have `fields=["raw_sum"]`, so each job yields 1 companion + 1 trace. Replace the final asserts with:

```python
    specs = Profiles.figures(res, params)
    assert len(specs) == 4  # 2 jobs × (companion + 1 trace)
    assert len({s.figure_id for s in specs}) == 4
    assert len({s.filename for s in specs}) == 4
```

`test_profiles_catalog_build_returns_figure` (near line 1242) — replace the `assert len(specs) == 1` line with `assert len(specs) == 3` (companion is `specs[0]`; the rest of the test — building `specs[0]` and asserting `>= 2` axes — is unchanged).

In `tests/test_export_fidelity.py`:

`test_profiles_export_disambiguates_shared_fig_name` (near line 186) — each job has `fields=["raw"]`, so 2 companions + 2 traces = 4 specs. Replace the final asserts with:

```python
    specs = Profiles.figures(result, params)
    stems = [s.filename for s in specs]
    assert len(stems) == 4  # 2 companion + 2 trace
    assert len(set(stems)) == 4, f"two jobs share an export stem -> silent overwrite: {stems}"
```

- [ ] **Step 2: Add the new catalog tests**

Append to `tests/test_figures_catalog.py` (after `test_profiles_catalog_build_returns_figure`):

```python
def test_profiles_catalog_traces_off_only_companion(tmp_path):
    """save_traces=False → only the companion spec per job."""
    import json

    h5 = tmp_path / "oblique_slices.h5"
    _write_profiles_h5(h5)
    job = _profiles_job_spec()
    res = _profiles_result(str(h5))
    params = {"consolidated_h5": str(h5), "jobs_json": json.dumps([job]), "save_traces": False}
    specs = Profiles.figures(res, params)
    assert len(specs) == 1
    assert specs[0].figure_id == "profiles_oblique_full"


def test_profiles_catalog_companion_off_only_traces(tmp_path):
    """save_companion=False → only the per-field trace specs."""
    import json

    h5 = tmp_path / "oblique_slices.h5"
    _write_profiles_h5(h5)
    job = _profiles_job_spec()
    res = _profiles_result(str(h5))
    params = {"consolidated_h5": str(h5), "jobs_json": json.dumps([job]), "save_companion": False}
    specs = Profiles.figures(res, params)
    assert len(specs) == 2
    assert all("__trace__" in s.figure_id for s in specs)


def test_profiles_catalog_trace_build_returns_single_axes_figure(tmp_path):
    """A trace spec's build(None) returns a single-axes Figure."""
    import json

    from matplotlib.figure import Figure

    h5 = tmp_path / "oblique_slices.h5"
    _write_profiles_h5(h5)
    job = _profiles_job_spec()
    res = _profiles_result(str(h5))
    params = {"consolidated_h5": str(h5), "jobs_json": json.dumps([job]), "save_companion": False}
    specs = Profiles.figures(res, params)
    fig = specs[0].build(None)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1  # one trace panel, no colorbar
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_figures_catalog.py -q -k profiles`
Expected: FAIL — the updated count assertions fail (`figures()` still emits only the companion) and the new tests fail.

- [ ] **Step 4: Extend `figures()`**

In `dfxm/stages/profiles.py`, in `figures()`, after the `line_override = ...` line (near line 764) add:

```python
    defaults = STAGE.defaults()
    save_companion = bool(params.get("save_companion", defaults["save_companion"]))
    save_traces = bool(params.get("save_traces", defaults["save_traces"]))
    trace_aspect = params.get("trace_aspect", defaults["trace_aspect"])
    trace_width_in = float(params.get("trace_width_in", defaults["trace_width_in"]))
    trace_linewidth = float(params.get("trace_linewidth", defaults["trace_linewidth"]))
    trace_color = params.get("trace_color", defaults["trace_color"]) or None
    trace_font_scale = float(params.get("trace_font_scale", defaults["trace_font_scale"]))
```

Then, inside the `for jr in jobs_with_fields:` loop, the current tail appends a single companion spec:

```python
        specs.append(
            FigureSpec(
                figure_id=f"profiles_{name}",
                title=f"Profile: {name}",
                kind="plot",
                filename=fig_name,
                build=_build,
            )
        )
```

Replace that block with (guard the companion by the toggle, then add trace specs):

```python
        if save_companion:
            specs.append(
                FigureSpec(
                    figure_id=f"profiles_{name}",
                    title=f"Profile: {name}",
                    kind="plot",
                    filename=fig_name,
                    build=_build,
                )
            )
        if save_traces:
            for vid in jr.fields:

                def _tbuild(
                    style,
                    *,
                    _h5=h5_path,
                    _job=job_spec,
                    _p=dict(params),
                    _ref=ref_pref,
                    _res=restrict,
                    _vid=vid,
                    _asp=trace_aspect,
                    _w=trace_width_in,
                    _lw=trace_linewidth,
                    _col=trace_color,
                    _fs=trace_font_scale,
                    _name=name,
                ):
                    if not _job:
                        raise ValueError(
                            f"job spec for {_name!r} not found in jobs_json — "
                            "re-run profiles with the current jobs_json to rebuild this figure"
                        )
                    if not _h5 or not os.path.exists(_h5):
                        raise FileNotFoundError(
                            f"consolidated h5 not found at {_h5!r} — re-run the slices stage"
                        )
                    import h5py as _h5py

                    _p.setdefault("geom_tol_um", STAGE.defaults().get("geom_tol_um", 1e-4))
                    _p.setdefault("offset_tol_um", STAGE.defaults().get("offset_tol_um", 1e-3))
                    with _h5py.File(_h5, "r") as f:
                        _, _fields, _geom, _ = _collect(f, _job, _p, _ref, _res)
                    _fld = next((fl for fl in _fields if fl["vid"] == _vid), None)
                    if _fld is None:
                        raise ValueError(f"field {_vid!r} not present for job {_name!r}")
                    return build_trace_figure(
                        _fld,
                        _geom,
                        aspect_wh=parse_aspect(_asp),
                        width_in=_w,
                        linewidth=_lw,
                        color=_col,
                        font_scale=_fs,
                        style=style,
                    )

                specs.append(
                    FigureSpec(
                        figure_id=f"profiles_{name}__trace__{vid}",
                        title=f"Profile trace: {name} · {vid}",
                        kind="plot",
                        filename=f"{fig_name}__trace__{vid}",
                        build=_tbuild,
                    )
                )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_figures_catalog.py tests/test_export_fidelity.py -q -k "profiles"`
Expected: PASS (updated + new profiles catalog tests, and the export-fidelity disambiguation test).

- [ ] **Step 6: Update `docs/Codebase.md`**

Replace the `figures(result, params)` bullet (line 365) with:

```markdown
- `figures(result, params)` — `@register("profiles")` catalog. Per parameter-mode job it emits a companion `kind="plot"` `FigureSpec` (`profiles_<name>`, rebuilt via `build_companion_figure`) when `save_companion`, followed by one trace `FigureSpec` per field (`profiles_<name>__trace__<vid>`, `filename=<fig_name>__trace__<vid>`, rebuilt via `build_trace_figure`) when `save_traces`. Toggles and the `trace_*` appearance params are read from `params` with a `STAGE.defaults()` fallback (both toggles default on). Each build re-reads `oblique_slices.h5` and re-`_collect`s the job. The export stem comes from each job's free-form `fig_name`; jobs that share a `fig_name` are disambiguated (`_2`, `_3`, …) and the trace stems inherit the disambiguated `fig_name`, so a batch export can't silently overwrite one figure with another.
```

Update the profiles data-flow row (line 471) from:

```markdown
| `profiles` | `oblique_slices.h5` | companion figures + CSVs + overviews | companion figure per parameter-mode job |
```

to:

```markdown
| `profiles` | `oblique_slices.h5` | per-field trace figures (+ optional companion) + CSVs + overviews | one trace `FigureSpec` per field (+ optional companion) per parameter-mode job |
```

- [ ] **Step 7: Update `docs/Usage.md`**

Update the catalog table row (line 684) from:

```markdown
| `profiles` | One companion figure per parameter-mode job (`kind="plot"`) — reference image + per-field line traces |
```

to:

```markdown
| `profiles` | Per parameter-mode job: one line-profile figure per field (`kind="plot"`), plus the stacked companion (reference image + all traces) when `save_companion` is on |
```

- [ ] **Step 8: Run ruff and commit**

Run: `ruff check dfxm/stages/profiles.py tests/test_figures_catalog.py tests/test_export_fidelity.py`
Expected: no errors.

```bash
git add dfxm/stages/profiles.py tests/test_figures_catalog.py tests/test_export_fidelity.py docs/Codebase.md docs/Usage.md
git commit -m "feat(profiles): register per-field trace FigureSpecs in publication export"
```

---

### Task 4: GUI results summary + preview picker

**Files:**
- Modify: `gui/stage_view.py` (`_summarize_profiles`, `_image_profiles`)
- Test: `tests/test_stage_summaries.py`
- Docs: `docs/Codebase.md` (stage_view note — optional one-liner)

**Interfaces:**
- Consumes: `ProfileJobResult.traces` / `.figure` (Task 2).
- Produces: `_summarize_profiles` lists `traces=N` and renders a `None` companion as `(no companion)`; `_image_profiles` returns the first job's `figure`, else the first job's first `traces` entry, else `None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stage_summaries.py` (near the other profiles tests):

```python
def test_representative_image_profiles_falls_back_to_trace():
    result = ProfilesResult(
        jobs=[ProfileJobResult(name="j", offset_used_um=0.0, figure=None, traces=["t.png"])]
    )
    assert _representative_image("profiles", result) == "t.png"


def test_summarize_profiles_reports_trace_count_and_no_companion():
    result = ProfilesResult(
        output_dir="o",
        mode="parameter",
        jobs=[
            ProfileJobResult(
                name="line1", offset_used_um=0.5, figure=None, traces=["a.png", "b.png"]
            )
        ],
    )
    out = _summarize("profiles", result)
    assert "traces=2" in out
    assert "(no companion)" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_summaries.py -q -k "profiles"`
Expected: FAIL — fallback returns `None`; summary has neither `traces=2` nor `(no companion)`.

- [ ] **Step 3: Update the helpers**

In `gui/stage_view.py`, replace `_summarize_profiles` (near line 699):

```python
def _summarize_profiles(result) -> str:
    lines = [
        f"mode: {result.mode}",
        f"output: {result.output_dir}",
        f"jobs: {len(result.jobs)}",
    ]
    for j in result.jobs:
        bits = []
        if j.csvs:
            bits.append(f"csv={len(j.csvs)}")
        if j.overviews:
            bits.append(f"overviews={len(j.overviews)}")
        if j.traces:
            bits.append(f"traces={len(j.traces)}")
        extra = (" " + " ".join(bits)) if bits else ""
        fig = j.figure or "(no companion)"
        lines.append(f"  {j.name} @ {j.offset_used_um:+.2f} µm -> {fig}{extra}")
    lines += [f"skipped: {s}" for s in result.skipped]
    return "\n".join(lines)
```

Replace `_image_profiles` (near line 777):

```python
def _image_profiles(result) -> str | None:
    for j in result.jobs:
        if j.figure:
            return j.figure
    for j in result.jobs:
        if j.traces:
            return j.traces[0]
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_summaries.py -q -k "profiles"`
Expected: PASS (fallback + summary tests, and the existing `..._uses_job_figure` still green).

- [ ] **Step 5: Update `docs/Codebase.md` (optional one-liner)**

In the `stage_view.py` row (line 403), the summarizer/picker are described generically ("one formatter per stage"); add a short clause to the profiles-relevant sentence only if a specific mention exists. If there is no profiles-specific sentence there, no doc change is required for this task (the generic description already covers it) — skip Step 5 rather than inventing text.

- [ ] **Step 6: Run the full suite, ruff, and commit**

Run: `python3 -m pytest -q`
Expected: PASS (whole suite; previously ~358 passed / 13 skipped, now higher with the added tests).

Run: `ruff check . && ruff format --check .`
Expected: clean.

```bash
git add gui/stage_view.py tests/test_stage_summaries.py
git commit -m "feat(gui): profiles results summary/preview handle per-field traces + no-companion"
```

- [ ] **Step 7: Final verification (verify-suite)**

Run the project's green-suite check before declaring done:

```bash
python3 -m pytest -q
ruff check . && ruff format --check .
python3 tests/gui_smoke.py
```

Expected: pytest green, ruff clean, `gui_smoke` steps `[1]`-`[26]` all OK.

---

## Self-Review

**Spec coverage:**
- Separate per-field trace figures → Task 1 (`build_trace_figure`, `_save_traces`) + Task 2 (`run()` writes them) + Task 3 (`figures()` registers them). ✔
- User-set aspect ratio (free-text `W:H`) → `parse_aspect` (Task 1) + `trace_aspect` param (Task 2). ✔
- Trace width → `trace_width_in` (Task 2), used in `build_trace_figure` (Task 1). ✔
- Line thickness → `trace_linewidth` (Task 2 param, Task 1 rendering). ✔
- Curve colour override (line + band, blank → `C0`) → `trace_color` (Task 2 param, Task 1 rendering + tests). ✔
- Trace-specific font scale, independent of map `font_scale` → `trace_font_scale` (Task 2 param, Task 1 rendering with `10*fs`/`12*fs`). ✔
- Companion behind a toggle, look unchanged → `save_companion` (Task 2), `build_companion_figure` untouched; existing regression tests only updated for spec counts, not for companion appearance. ✔
- Publication-export parity (traces in `figures()`) → Task 3. ✔
- GUI summary/preview handle new outputs → Task 4. ✔
- Docs updated in-change → Usage.md/Codebase.md steps in Tasks 1-3 (and generic-only for Task 4). ✔
- Tests: param metadata (Task 2 runs it), aspect parse (Task 1), render/aspect/linewidth/color (Task 1), toggles (Task 2), figures() specs (Task 3), catalog test updates (Task 3), GUI fallback (Task 4). ✔

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test shows real assertions. Task 4 Step 5 is explicitly conditional-skip (not a placeholder — it instructs to skip rather than invent doc text). ✔

**Type consistency:** `parse_aspect(s) -> tuple[float,float]`, `build_trace_figure(..., aspect_wh, width_in, linewidth, color, font_scale, style)`, `_save_traces(..., aspect, width_in, linewidth, color, font_scale, dpi, style)`, `ProfileJobResult.traces: list[str]`, `figure_id`/`filename` patterns `profiles_<name>__trace__<vid>` / `<fig_name>__trace__<vid>` — used consistently across Tasks 1-4. `figures()` reads toggles/params with `STAGE.defaults()` fallback, matching `run()`'s defaulting. ✔
