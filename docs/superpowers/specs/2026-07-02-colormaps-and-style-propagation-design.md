# Per-group colormaps + publication style propagation to runs — design

Date: 2026-07-02
Status: approved (design review with Albert, this session)

## Problem

1. **Mosa CoM renders with the wrong colormap.** The slices stage requests
   `"fast"` (ParaView's default *Fast* map) for mosaicity centre-of-mass, but
   `get_cmap` in `dfxm/common/plotting.py` silently falls back to matplotlib's
   `coolwarm` because *Fast* is not registered — visually a blue-white-red map
   that reads as `RdBu_r`. Stages are also inconsistent: visualize uses `magma`
   for mosa CoM, slices uses `fast`, rocking renders raw intensity with `magma`
   while slices uses `gray` for the same quantity.
2. **No user control of colormaps.** There is no dropdown anywhere to choose
   the colormap per quantity group (mosa CoM, mosa FWHM, strain, raw).
3. **Publication style edits never reach runs.** The "Publication style…"
   dialog mutates the session `PlotStyle`, but only **Export/Export-all**
   consume it. PNGs written *during a stage run* are hard-wired to the legacy
   style (`style=None`), so editing the style and re-running changes nothing.
4. **Style is session-only.** The edited style is lost on app restart.

## Decisions (user-approved)

- Colormap dropdowns apply to **everything that plots** (slices, visualize,
  rocking, mosaicity maps, strain maps/diagnostics; profiles inherits from
  `oblique_slices.h5`; matched keeps its existing explicit `colormap` param).
- Group defaults, consistent across all stages: **mosa CoM = fast** (real
  ParaView map, newly registered), **mosa FWHM = magma**, **strain = RdBu_r**,
  **raw = gray**.
- **Runs always use the current publication style** — snapshotted at Run
  click; no confirm button.
- Style (including the colormap choices) **persists via QSettings**.

## Design

### 1. Register the real ParaView *Fast* colormap

New module `dfxm/common/cmaps.py`:

- Holds ParaView's authoritative *Fast* definition (9 RGB control points,
  `ColorSpace: "Lab"`, creators Samsel & Scott — fetched from
  `Remoting/Views/ColorMaps.json` on ParaView master, 2026-07-02):

  ```
  0.0                 0.0564  0.0564  0.4700
  0.17159223942480895 0.2430  0.46035 0.8100
  0.2984914818394138  0.35681 0.74502 0.95437
  0.4321287371255907  0.6882  0.9300  0.91791
  0.5                 0.89950 0.94465 0.76866
  0.5882260353170073  0.95711 0.83382 0.50892
  0.7061412605695164  0.92752 0.62144 0.31536
  0.8476395308725272  0.8000  0.3520  0.1600
  1.0                 0.5900  0.0767  0.11947
  ```

- Builds a 256-sample `ListedColormap` by interpolating between control
  points in **CIELAB** space (sRGB → Lab, linear interp on the normalized
  positions, Lab → sRGB, clip to [0,1]) — matching how ParaView renders it.
  The Lab round-trip is ~30 lines of numpy (D65, standard sRGB transfer);
  no new dependency.
- `register()` registers it with `matplotlib.colormaps` under the name
  `"fast"`; called once at `dfxm/common/plotting.py` import. Because it is a
  normal registered matplotlib colormap, pyvista 3-D viewers resolve the name
  too.
- `get_cmap`'s `fast → coolwarm` fallback is **removed** (keep `get_cmap` as
  the lookup helper; it now finds `"fast"` in the registry).

### 2. Colormap groups on `PlotStyle`

- `PlotStyle` gains four fields:
  `cmap_mosa_com: str = "fast"`, `cmap_mosa_fwhm: str = "magma"`,
  `cmap_strain: str = "RdBu_r"`, `cmap_raw: str = "gray"`,
  plus `cmap_for(group: str) -> str` where group ∈
  `{"mosa_com", "mosa_fwhm", "strain", "raw"}` (unknown group raises).
- `StyleControls` (used by both the global **Publication style…** dialog and
  every stage's **Export…** dialog — i.e. the dropdowns appear in each
  plotting part of the program) gains a **Colormaps** section: four dropdowns
  with the curated list
  `fast, magma, viridis, plasma, inferno, cividis, gray, bone, RdBu_r,
  coolwarm, seismic, turbo`.
- Figure builders resolve their colormap from the active style **at build
  time** (so changing the dropdown in an export dialog re-renders correctly):
  - visualize: `_display_info` returns a *group* for CoM/FWHM/strain;
    layer-figure calls and `volume_layer_specs` closures resolve via
    `style.cmap_for(group)`.
  - mosaicity: `_KEY_DISPLAY` CoM keys → `mosa_com` group, FWHM keys →
    `mosa_fwhm`; unknown keys keep their current fixed fallback (`magma`).
  - rocking: raw volumes (sum / specific frame / per-angle) → `raw` group,
    in both run-time rendering and figure specs.
  - strain: strain map + detrend diagnostic imshows → `strain` group. The
    histogram (no colormap) is untouched.
  - slices: `_STD_VOLUMES` drops its hard-coded cmap column; the volume
    *kind* maps to the group (`raw_sum`/`raw_specific` → `raw`). The
    **resolved** name is written to the `cmap` attr in `oblique_slices.h5`,
    so profiles and the GUI line picker inherit with zero changes. At export
    time, slices figure builders re-resolve from the style using the stored
    `kind`, falling back to the stored `cmap` attr for old files.
  - matched: keeps its explicit `colormap` param (already user-settable);
    if it is a plain STR param it becomes an ENUM with the same curated list.
- When no style is available (`style=None`, headless CLI), builders use
  `PlotStyle()` field defaults for the group — i.e. the defaults above.
  Visible default changes: visualize mosa CoM `magma → fast`, rocking raw
  layers `magma → gray`, slices mosa CoM `coolwarm(fallback) → real fast`.

### 3. Publication style flows into every run

- `StageView._on_run` injects a reserved, undeclared key into the params dict
  it hands the runner: `params["plot_style"] =
  dataclasses.asdict(self.window().global_plot_style())`. It is not a
  `Param`, never appears in forms, and is ignored by validation.
- Each stage's `run()` reconstructs it once:
  `style = PlotStyle(**{k: v for k, v in p["plot_style"].items() if k in
  fields}) if p.get("plot_style") else None` via a small shared helper
  `style_from_params(p)` in `dfxm/common/plotting.py` (tolerant of missing /
  unknown keys for forward compatibility; `formats` list → tuple).
- All run-time render paths thread that style instead of implying legacy:
  - `render.save_layer_pngs` / `save_layer_animation` gain a `style=None`
    kwarg passed through to `layer_figure` (visualize + rocking callers).
  - slices `save_slice_png` passes the style to `build_slice_figure`.
  - profiles run-time companion + overview figures accept the style (same
    styled primitives as their export builders).
  - matched run-time layer figures pass the style to `Rnd.layer_figure`.
  - strain diagnostics: strain map / detrend / histogram builders accept the
    style where the shared primitives apply (colormap + font scale at
    minimum; these figures keep their bespoke layouts otherwise).
- Behavior change (approved): run outputs are publication-styled by default
  (seeded style is `PUBLICATION_STYLE`). Headless CLI runs (no `plot_style`
  key) keep today's legacy look exactly.

### 4. Persistence

- `MainWindow` saves the style to `QSettings` (key `plot_style`, JSON of
  `asdict`) when the Publication-style dialog closes and in `closeEvent`;
  loads it in `__init__` (missing → seeded from `PUBLICATION_STYLE`; unknown
  keys dropped, missing keys defaulted — an old settings blob never crashes a
  newer app).

## Out of scope

- Per-stage colormap overrides in stage parameter forms (rejected in favour
  of style-level dropdowns; revisit only if a real need appears).
- Restyling the mosaicity RGB composite / KAM figures beyond the four groups.
- The two deferred plot-export follow-ups (altitude refactors) stay deferred.

## Testing

- `tests/test_cmaps.py`: `"fast"` resolves via `matplotlib.colormaps` and
  `get_cmap`; endpoints/midpoint match the ParaView control points to a few
  1e-3; the coolwarm fallback is gone (registry-backed).
- `PlotStyle.cmap_for` mapping + `style_from_params` round-trip (including
  tolerance of unknown keys and missing keys).
- Per-stage: builders honour a style with a non-default group cmap (assert
  the imshow's colormap name), for visualize, rocking, slices, strain,
  mosaicity spec builders.
- Slices: `oblique_slices.h5` `cmap` attr equals the style-resolved name;
  profiles reference figure uses it.
- Run injection: a stage run given `plot_style` with e.g. `font_scale=3`
  and `cmap_raw="viridis"` produces PNGs whose figure reflects both (unit
  test on the render path, plus `gui_smoke` step asserting `_on_run` injects
  the key).
- QSettings round-trip: save → load reproduces the style; corrupted/legacy
  blob falls back to defaults.
- Existing suites stay green (`python3 -m pytest -q`, `ruff`, `gui_smoke`).

## Docs

- `docs/Usage.md`: new "Colormaps & publication style" subsection (where the
  dropdowns live, the four groups, runs-use-current-style, persistence);
  update stage sections whose default appearance changes.
- `docs/Codebase.md`: `dfxm/common/cmaps.py` module entry; `PlotStyle`
  fields + `cmap_for`/`style_from_params`; the reserved `plot_style` params
  key; render/stage signature changes.
