# GUI overhaul — design spec

**Date:** 2026-06-11
**Status:** approved in brainstorming; awaiting final review
**Scope:** the DFXM pipeline GUI (`gui/`) plus the Qt-free schema that drives it (`dfxm/config/models.py`, stage `StageSpec`s)

## Context & goals

The current GUI shows every stage parameter in one flat form (slices: 27, paraview: 26,
rocking: 25), help text exists for only ~35 % of params and only as hover tooltips, and
the left column stacks a full experiment form plus two redundant stage lists. The
overhaul makes the app usable by **first-time beamline users** — people who may not know
DFXM concepts well — without slowing down experts.

Goals, as agreed:

1. **Decluttered forms** — each stage shows 4–7 essential fields; everything else sits in
   one collapsed *Advanced* expander, grouped under themed headers.
2. **Full help coverage** — every parameter (~168 stage params + the experiment schema)
   gets help text written for a first-timer, physics included, surfaced in a
   focus-following help panel plus hover tooltips.
3. **Reorganized window** — a single pipeline rail with built-in status, a compact
   experiment header with an Edit dialog, and an Overview landing page. darfix appears
   as an external step; **concat is visibly optional**.
4. **Friendlier run feedback** — plain-language errors with actionable hints, pre-run
   input validation, and a visible progress bar.

## Non-goals

- Publication-quality plot export (scale bars, font scaling, preview) — that is the
  second project and gets its own spec.
- Results/Output tab redesign (thumbnails, folder links) — explicitly excluded.
- Generating stage CLIs from the specs — the hand-written argparse CLIs stay as-is.
- Persisting run status/history across sessions — status remains per-session.
- New stage functionality or changes to any computation.

## Approach

Extend the schema, keep the GUI generated (chosen over GUI-side curation tables and
hand-built per-stage panels). All declutter/help metadata lives in the stage specs —
single source of truth, the GUI keeps auto-building forms, new stages inherit the UX,
and `dfxm/` stays Qt-free.

## Design

### 1. Schema & help content (Qt-free core)

`Param` (`dfxm/config/models.py`) gains three backwards-compatible fields:

- `advanced: bool = False` — `True` moves the param into the Advanced expander.
  Params left `False` are the stage's essentials, shown at the top in spec order.
- `group: str = ""` — themed section header inside the Advanced expander
  (e.g. "Calibration", "Data layout"). Required when `advanced=True`; unused otherwise.
- `must_exist: bool = False` — set on `PATH`/`DIR` params that are *inputs* (read, not
  created, by the stage). Drives the GUI's pre-run validation; never set on output
  locations, which stages create themselves.

**Help content.** Every `Param` in all 10 stage specs and in `EXPERIMENT_SCHEMA` gets a
`help` string aimed at a first-time beamline user: what the field does, its unit, when
you would change it, with the physics explained where needed. Calibration params
additionally state the consequence of a wrong value (e.g. "a wrong reference angle
silently shifts every strain map"). Drafted from `docs/Usage.md` and the code; the
calibration/physics texts are flagged for domain review before merge.

**Stage descriptions.** `StageSpec.description` (existing field) is rewritten per stage
as 1–2 newcomer-friendly sentences: what the stage consumes, what it produces, where it
sits in the flow. Feeds the Overview page and the help panel's idle text.

**Enforcement test.** A new pytest asserts, for every stage in
`dfxm.stages.registry.STAGE_TARGETS`: every param has non-empty `help`; every
`advanced=True` param has a non-empty `group`; and every stage has ≤ 8 essentials.
Same spirit as the existing `_SUMMARIZERS` sync test — future stages stay honest.

**Group vocabulary** (shared across stages, so users learn it once):

| Group | Contents |
|---|---|
| Calibration | pixel scales, reference angles — pre-filled from the experiment, ⚠-flagged |
| Data layout | HDF5 internal paths, filename patterns, entry suffixes — pre-filled, rarely touched |
| Alignment | ROI, centring method, samy direction, tolerances |
| Appearance | colour limits, colormaps, opacity, DPI |
| Output | filenames, formats, what-to-save toggles |

Stage-specific groups where needed: *Export* (paraview), *Quantities* (slices),
*Selection* and *Matching* (profiles).

### 2. GUI widgets

**`gui/widgets/param_form.py` — grouped form.** Essentials render as a normal form
layout at the top, in spec order. Below them, one collapsed
**"▸ Advanced (N settings)"** expander (arrow-style `QToolButton` toggling a container).
Inside it, bold group headers with each group's rows beneath, groups in
first-appearance order. Public API unchanged (`values()`, `set_values()`, `changed`),
so `StageView` and the experiment editor are untouched by the internals. Calibration
styling (red label + ⚠) kept. New signal `focusedParamChanged(Param)`, emitted via an
event filter installed on every editor widget when it gains focus.

**`gui/widgets/help_panel.py` — focus-following help (new).** A styled read-only box
pinned beneath the form on the stage panel's left side. Shows the focused param's
label, unit, ⚠-calibration warning, and full help text; shows the stage description
when nothing is focused. Listens to `focusedParamChanged`. Hover tooltips remain.

**`gui/experiment_panel.py` — compact header.** One line: preset dropdown, muted
summary of key calibration values ("ccmth 10.65° · 0.203 µm/px"), reload button, and
**Edit…**. The full field form, notes display, Apply, and Save-as move into a modal
dialog reusing `ParamForm` + the help panel (experiment fields get help texts too).
The `experimentChanged` signal contract is unchanged, so stage pre-fill keeps working.

**Pipeline rail (`gui/main_window.py`) — replaces the nav list + status list.** A
single list: **Overview** at top, then the stages in pipeline order, numbered, each
with a status glyph (— idle, ▶ running, ✓ ok, ✗ failed). Two special renderings:

- **Concat** renders as "Concat *(optional)*" in muted style — skippable if scans are
  already merged; the Overview page says so explicitly.
- **darfix** appears between concat and strain as a greyed, *non-selectable*
  "⤷ darfix (external)" row with a tooltip explaining it runs outside the app.

**`gui/overview_page.py` — landing page (new).** Default screen on launch; first item
in the stack. Contents: the pipeline drawn left-to-right as clickable chips (concat
optional-styled, darfix dashed/external), the 1–2 sentence description per stage from
`StageSpec.description`, and a per-session status recap. Clicking a chip navigates to
that stage.

### 3. Run feedback

**Core (`dfxm/common/errors.py`, new).** `StageUserError(message, hint)` — raised by
stages at known choke points: input folder missing, glob matches nothing, `maps.h5`
absent ("run darfix on the concatenated file first"), no scans found. The runner
(`dfxm/runner.py`) catches it; the `Failed` message gains an optional `hint` field.
Unexpected exceptions keep today's behaviour (full traceback, empty hint). Stages adopt
`StageUserError` in their input-validation paths only — no computation changes.

**GUI banner (`gui/stage_view.py`).** A status banner above the tabs, hidden when
idle. On failure: red, plain-language message + hint, with a "show log" link that
switches to the Log tab (full traceback unchanged there). On success: brief green
one-liner. **Pre-run validation:** before launching, every *non-empty* param flagged
`must_exist` is checked on disk. A bad path blocks the run, shows the banner, and
focuses the field so the help panel explains it. Empty fields are deliberately left to
the stage's own `StageUserError` validation — which fields are required can depend on
`mode` (e.g. strain in batch mode ignores `input_folder`), and the stage knows that.

**Progress.** Stages already emit progress through the runner; the GUI currently
buries it in the log. Add a `QProgressBar` beside Run/Cancel showing percent (when
reported) and the latest step description; the rail glyph switches to ▶ while running.

## Per-stage essentials & groups (for domain review)

Proposed split. Essentials listed in display order; everything else goes to Advanced
under the named group. **Reviewer: adjust freely — this table is the contract.**

### concat (5 essentials / 14)
- **Essentials:** `mode`, `input_folder`, `root_folder`, `folder_pattern`, `skip_existing`
- **Data layout:** `h5_filename_override`, `entry_suffix`, `detector_read_path`, `detector_write_path`, `positioners_path`, `output_entry`
- **Output:** `vds_policy`, `copy_data`, `overwrite`

### strain (5 essentials / 15)
- **Essentials:** `mode`, `input_folder`, `root_folder`, `roi`, `output_dir`
- **Calibration:** `ccmth_ref_deg`, `pixel_size_x_um`, `pixel_size_y_um`
- **Data layout:** `folder_pattern`, `maps_filename`, `ccmth_com_path`
- **Appearance:** `vmin`, `vmax`, `save_plots`
- **Output:** `stacked_filename`

### mosaicity (4 essentials / 12)
- **Essentials:** `mode`, `input_folder`, `root_folder`, `output_dir`
- **Data layout:** `folder_pattern`, `maps_filename`, `chi_com_path`, `chi_fwhm_path`, `mu_com_path`, `mu_fwhm_path`
- **Output:** `stacked_filename`, `compression`

### visualize (6 essentials / 20)
- **Essentials:** `mosa_volume_file`, `strain_volume_file`, `raw_root`, `roi_x`, `roi_y`, `output_dir`
- **Calibration:** `pixel_size_x_um`, `pixel_size_y_um`
- **Data layout:** `mosa_pattern`, `strain_pattern`, `samy_path`, `samz_path`
- **Alignment:** `samy_direction`, `center_method`, `range_pct`
- **Appearance:** `volume_opacity`
- **Output:** `output_format`, `save_layers`, `save_animation`, `save_topview`

### rocking (5 essentials / 25)
- **Essentials:** `raw_root`, `roi_x`, `roi_y`, `specific_frame_idx`, `output_dir`
- **Calibration:** `pixel_size_x_um`, `pixel_size_y_um`
- **Data layout:** `rocking_pattern`, `mosa_pattern`, `strain_pattern`, `samy_path`, `samz_path`, `detector_path`
- **Alignment:** `samy_direction`, `samz_tol_mm`, `normalize_sum`
- **Appearance:** `volume_opacity`, `cbar_pct_lo`, `cbar_pct_hi`
- **Output:** `aligned_h5_name`, `save_aligned_h5`, `save_layers`, `save_animation`, `save_topview`, `output_format`

### paraview (6 essentials / 26)
- **Essentials:** `mosa_volume_file`, `strain_volume_file`, `raw_root`, `roi_x`, `roi_y`, `output_dir`
- **Calibration:** `pixel_size_x_um`, `pixel_size_y_um`
- **Data layout:** `mosa_pattern`, `strain_pattern`, `samy_path`, `samz_path`
- **Alignment:** `samy_direction`, `center_method`, `center_mosa_com`, `center_strain`, `abs_mosa_fwhm`, `anchor_origin_to_reference`, `mosa_darfix_origin_xy`, `strain_darfix_origin_xy`
- **Export:** `num_pieces_z`, `piece_compression`, `replace_nan`, `write_valid_mask`, `export_mosaicity`, `export_strain`

### slices (6 essentials / 27)
- **Essentials:** `mosa_volume_file`, `strain_volume_file`, `aligned_rocking_file`, `raw_root`, `slices_json`, `output_dir`
- **Calibration:** `pixel_size_x_um`, `pixel_size_y_um`
- **Data layout:** `mosa_pattern`, `strain_pattern`, `samy_path`, `samz_path`
- **Alignment:** `samy_direction`, `align_roi_x`, `align_roi_y`, `abs_fwhm`, `center_method`, `range_pct`
- **Quantities:** `include_mosa_com_chi`, `include_mosa_fwhm_chi`, `include_mosa_com_mu`, `include_mosa_fwhm_mu`, `include_strain`, `include_raw_sum`, `include_raw_specific`
- **Output:** `output_h5_name`, `save_png`

### profiles (4 essentials / 12)
- **Essentials:** `consolidated_h5`, `mode`, `jobs_json`, `output_dir`
- **Selection:** `reference_volume_id`, `volume_ids`
- **Matching:** `geom_tol_um`, `offset_tol_um`
- **Appearance:** `line_color`, `fig_dpi`
- **Output:** `save_csv`, `save_overview`

### matched (4 essentials / 17)
- **Essentials:** `raw_root`, `frame_index`, `match_threshold_mm`, `output_dir`
- **Calibration:** `pixel_size_x_um`, `pixel_size_y_um`
- **Data layout:** `strain_pattern`, `rocking_pattern`, `samy_path`, `samz_path`, `pco_ff_path`
- **Alignment:** `samy_direction`
- **Appearance:** `colormap`, `vmin`, `vmax`, `auto_pct_lo`, `auto_pct_hi`

## Testing

- **pytest (Qt-free):** the schema enforcement test (help coverage, group-on-advanced,
  essentials ≤ 8); unit tests for `StageUserError`/`Failed.hint` round-tripping through
  the runner; tests that representative stages raise `StageUserError` with useful hints
  on synthetic broken inputs (e.g. a layer folder without `maps.h5`).
- **Offscreen GUI (`tests/gui_smoke.py`, deliberately not pytest-collected):** build
  every stage's grouped form and read `values()` back; focus a field and assert the
  help panel text; navigate the rail including Overview and the non-selectable darfix
  row; run a deliberately failing stage and assert the banner shows message + hint;
  open and apply the experiment Edit dialog.

## Documentation updates (same change, per repo contract)

- `docs/Usage.md`: rewritten GUI tour (rail, Overview, optional concat, Advanced
  expander, help panel, banner, progress bar); each stage section lists its essentials.
- `docs/Codebase.md`: new modules (`help_panel`, `overview_page`, `errors`), reworked
  `param_form`/`experiment_panel`/`main_window`/`stage_view`, `Param.advanced`/`.group`,
  `Failed.hint`.
- `CLAUDE.md` "Adding a stage": params must set `help` (all) and `group` (advanced
  ones); write the newcomer `description`; the enforcement test fails otherwise.

## Implementation order (suggested phases)

1. Schema fields + `StageUserError` + `Failed.hint` (core, no behaviour change).
2. Help texts, `advanced`/`group`/`must_exist` metadata, and descriptions for all 10
   stages + the experiment schema; enable the enforcement test.
3. `ParamForm` grouping + `focusedParamChanged` + `HelpPanel`, wired into `StageView`.
4. Compact experiment header + Edit dialog; pipeline rail; Overview page.
5. Failure banner, pre-run path validation, progress bar; stage adoption of
   `StageUserError` in input-validation paths.
6. `gui_smoke` extensions; final docs pass.

Each phase keeps the app fully working; docs updates accompany the phase that changes
the behaviour they describe.
