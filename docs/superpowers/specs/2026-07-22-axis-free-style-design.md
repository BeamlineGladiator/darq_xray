# Axis-free plot style — design

Date: 2026-07-22
Status: approved (brainstormed with Albert, sections approved individually)
Origin: axis-free-plot-style wish (2026-07-17) — publication-style option to
remove the plot box/spines, or the x/y axes entirely.

## Goal

Add an **Axes** mode to the publication `PlotStyle` so styled map figures can be
exported without the plot frame, or without any axis decoration at all — the
axis-free map look where the scale bar and colorbar carry the physical context.

## Decisions (agreed during brainstorming)

- **One 3-mode dropdown**, not granular checkboxes and not a single all-or-nothing
  checkbox:
  - `full` (default) — today's look, unchanged output.
  - `no_frame` — the four spines hidden; ticks, tick labels and the x/y (µm)
    axis labels remain.
  - `none` — spines, ticks, tick labels and axis labels all removed.
- **Map figures only.** Profiles trace figures, the companion figure (both its
  map panel and its trace axes — the map panel shares the distance frame with
  the traces below it), the strain histogram, the strain detrend diagnostic,
  diagnostic PNGs and every legacy `style=None` path are untouched.
- **No coupling** between axes mode and the scale bar (or title/colorbar): all
  stay independent knobs. `show_title`, `colorbar`, `scale_bar` behave exactly
  as today under every mode.
- **Global style field** (session-wide `PlotStyle`), no per-stage override. The
  replot dialogs render with the same global style, so they pick the mode up
  automatically with zero replot-side changes.
- **Tight-crop export unchanged**: `bbox_inches="tight"` simply crops closer
  once decorations vanish — desired behaviour.

## Implementation

### Core — `dfxm/common/plotting.py`

- New `PlotStyle` field, grouped with the text knobs:

  ```python
  axes_mode: str = "full"  # "full" | "no_frame" | "none"
  ```

- New helper:

  ```python
  def apply_axes_mode(ax, style):
      if style.axes_mode == "no_frame":
          for sp in ax.spines.values():
              sp.set_visible(False)
      elif style.axes_mode == "none":
          ax.set_axis_off()
      # "full" and any stale/unknown persisted value: no-op (defensive,
      # same pattern as fixed_scale) — never raises.
  ```

- `PUBLICATION_STYLE` keeps `axes_mode="full"` — no default-output change.
- Serialization is free: `style_to_json`/`style_from_json`/`_style_from_dict`
  already default missing keys, so pre-existing QSettings snapshots load as
  `"full"`; `style_from_params` injection into stages needs no change.

### Call sites — paired with the existing `apply_text_scale` call, map axes only

| Site | Covers |
| --- | --- |
| `dfxm/common/render.py` (shared styled volume-layer renderer, ~line 79) | mosaicity, rocking, visualize, matched maps; replot `render_volume_layer`; animation frames |
| `dfxm/stages/strain.py` strain-map builder (~line 428) | strain map (NOT the detrend diagnostic at ~488, NOT the histogram) |
| `dfxm/stages/slices.py` plane figure (~line 872) | slices plane figures |
| `dfxm/stages/profiles.py` per-field map (~line 895) | profiles reference/field maps (NOT companion, NOT traces) |

Call order: immediately after `apply_text_scale(ax, style)`. The planning pass
must verify this call-site inventory is complete (rule: every styled *map*
builder that calls `apply_text_scale` on a map axes gets the paired call; no
non-map axes does).

### GUI — `gui/widgets/export_dialog.py` (`StyleControls`)

One `QComboBox` labelled **Axes** with display items Full / No frame / None
mapped to the three values, wired like every other control: mutates
`style.axes_mode`, emits `changed`, restored in `sync_from_style`. Placed with
the text controls (near Font scale / Show title). `main_window` persistence
(whole-style JSON in QSettings) needs no change.

## Edge cases

- Bogus/stale `axes_mode` value in a persisted style → treated as `full`
  (no-op), never raises.
- Fixed physical scale (`scale_um_per_cm` + `fit_axes_to_box`): hiding
  decorations does not move the axes box, so the µm-per-cm print size is
  unaffected; a test pins the axes bbox across modes.
- Animation frames inherit the mode via the shared renderer. ParaView/3-D
  outputs are not matplotlib figures — unaffected.
- No user-facing error surface; no `StageUserError` needed.

## Testing

- **Helper units:** `no_frame` → all four spines invisible, ticks/labels
  intact; `none` → `ax.axison` False; `full` and unknown values → no-op.
- **Round-trip:** `axes_mode` survives `style_to_json`/`style_from_json`; a
  legacy snapshot without the key loads as `full`.
- **Builders:** strain map + shared renderer honour the mode; detrend
  diagnostic, histogram, companion and trace figures keep their axes under
  `axes_mode="none"`; `style=None` legacy path unchanged.
- **Fixed-scale fidelity:** axes bbox position identical across modes.
- **gui_smoke step:** open the Publication style dialog, switch Axes to None,
  assert `style.axes_mode == "none"` and QSettings persistence round-trips.

## Documentation (same change, per contract)

- `docs/Usage.md` — Publication-style section: the Axes dropdown, its three
  modes, maps-only scope (traces/companion keep axes), and a tip that under
  None the scale bar is the only spatial reference.
- `docs/Codebase.md` — `plotting.py` entry gains `axes_mode` +
  `apply_axes_mode`; call-site notes under render/strain/slices/profiles and
  the `export_dialog.py` entry.
