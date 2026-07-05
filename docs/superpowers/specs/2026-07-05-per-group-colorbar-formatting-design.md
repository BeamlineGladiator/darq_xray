# Per-group colourbar number formatting — design

Date: 2026-07-05
Status: approved (brainstorm), pending implementation plan

## Motivation

The publication export style currently controls colourbar number formatting with
a **single** `PlotStyle.colorbar_tick_format` field applied to every figure. Three
problems, all raised by the user:

1. **Scientific-notation offset text (`×10ⁿ`) is unstyleable.** It sits at the top
   of the vertical colourbar and, on a real matplotlib bug, does *not* grow with
   `font_scale` — so at large fonts the exponent is a tiny sliver. The top
   placement also collides with the title, which the code works around with the
   `_TITLE_PAD_PER_FONT_SCALE` hack in `apply_text_scale`.
2. **Formatting can't differ per quantity.** Strain (values ~1e-4) wants
   scientific notation; mosaicity angles want plain decimals; raw intensity wants
   neither. One global field can't express that.
3. **Raw intensity has no meaningful absolute scale.** The user wants raw
   colourbars to drop numeric ticks entirely and instead be marked "arbitrary
   units".

## Decisions (from brainstorming Q&A)

- **Arbitrary units** is exposed as a new *value* in the per-group tick-format
  dropdown (`"arb"`), not a separate checkbox. Reusable for any group; used for raw.
- **Offset-text position** is user-selectable top/bottom; the Publication preset
  defaults to **bottom**.
- **Offset-text size and position are per group** (not global), alongside the
  per-group tick-format choice.
- **Default tick format is tuned per group**: strain → scientific, mosaicity
  COM & FWHM → auto, raw → arbitrary units. Applies to fresh styles and to
  legacy persisted styles that predate this change.

## Scope of quantity groups

The four existing groups drive per-group colormaps already
(`CMAP_GROUPS = ("mosa_com", "mosa_fwhm", "strain", "raw")`) and the group is
determinable at **every** `add_colorbar` call site (each one already computes a
colormap via `resolve_cmap(style, group)` or an equivalent kind→group map). This
design reuses that exact plumbing — no new group taxonomy.

## Component 1 — `PlotStyle` schema (`dfxm/common/plotting.py`)

Remove the single `colorbar_tick_format` field. Add three families of per-group
fields (12 total), named to match the existing `cmap_<group>` convention:

```
tickfmt_mosa_com / _mosa_fwhm / _strain / _raw   : str    default "auto"
offset_scale_mosa_com / ... / _raw               : float  default 1.0
offset_pos_mosa_com / ... / _raw                 : str    default "top"
```

Add lookup helpers mirroring `cmap_for`:

```python
def tickfmt_for(self, group: str | None) -> str      # None -> "auto"
def offset_scale_for(self, group: str | None) -> float  # None -> 1.0
def offset_pos_for(self, group: str | None) -> str   # None -> "top"
```

`group=None` is a valid input (an `add_colorbar` caller that does not know its
group) and returns the neutral defaults, so the function never raises for a
missing group. Unknown non-None groups still raise `KeyError` like `cmap_for`.

Tick-format value set: `"auto"`, `"scientific"`, `"arb"`, and digit strings
`"0".."3"` (unchanged plain-decimal behaviour).

### Defaults and the legacy-render contract

- **Bare `PlotStyle()` keeps `auto` / `1.0` / `top` for all groups.** The
  `style=None` code path (`render.layer_figure` fallback, headless CLI) builds a
  bare `PlotStyle` and MUST stay byte-identical with today's output. `auto` →
  matplotlib default → no change. This is the invariant that the existing
  "legacy path pinned by regression test" work depends on.
- **`PUBLICATION_STYLE`** sets the tuned profile explicitly:
  `tickfmt_strain="scientific"`, `tickfmt_raw="arb"`, mosa groups `"auto"`;
  all `offset_pos_*="bottom"`; all `offset_scale_*=1.0`.
- **Migration for old serialized styles.** `_style_from_dict` is only ever
  reached from `style_from_json` (QSettings persistence) and `style_from_params`
  (stage-injected GUI style) — never from the bare-`PlotStyle` code path. When the
  incoming dict contains **none** of the four `tickfmt_*` keys, it is an
  old snapshot: inject the tuned profile (same as `PUBLICATION_STYLE`:
  strain→scientific, raw→arb, mosa→auto, offset_pos→bottom). A current GUI always
  serializes all 12 fields via `asdict`, so new dicts never trigger the
  migration. The obsolete `colorbar_tick_format` key, if present, is ignored
  (dropped by the existing key-filter).

## Component 2 — `add_colorbar` rendering (`dfxm/common/plotting.py`)

New signature: `add_colorbar(fig, im, ax, label, style, *, group=None)`.
`group` is keyword-only with a default so existing 5-positional-arg callers and
tests keep working (they resolve to the neutral `auto` behaviour).

Branch on `fmt = style.tickfmt_for(group)`:

- **`auto` / `"0".."3"`** — unchanged: `_tick_formatter(fmt)` returns matplotlib
  default (`None`) or a fixed-decimal `FuncFormatter`. Tick label + colourbar
  label sizes scale by `font_scale` exactly as today.

- **`scientific`** — unified custom-exponent renderer (chosen over repositioning
  matplotlib's built-in offset, which is fragile because
  `YAxis._update_offset_text_position` re-derives that artist's position on every
  draw):
  1. Compute one common exponent from the colour limits:
     `oom = 0 if maxabs == 0 else floor(log10(maxabs))`, `maxabs = max(|vmin|,|vmax|)`.
  2. Format ticks as mantissas via a `FuncFormatter` (`value / 10**oom`, small
     fixed number of decimals).
  3. Hide matplotlib's built-in offset text
     (`cb.ax.yaxis.get_offset_text().set_visible(False)`).
  4. Draw our own exponent label as a `Text` on the colourbar axes at the
     group's `offset_pos` (top → just above the bar, `va="bottom"`; bottom → just
     below, `va="top"`) with fontsize `9 * font_scale * offset_scale`. Redraw-safe
     (a static artist), independently sizeable, and freely positionable.
  5. When `oom == 0` the exponent label is suppressed (mantissas are the values
     themselves).

- **`arb`** — arbitrary units: `cb.set_ticks([])` (drops all numeric ticks,
  overriding the global `colorbar_ticks` count for this group) and append
  " (arb. units)" to the colourbar label. The suffix is skipped when the label
  already mentions arbitrariness (`"a.u."` / `"arb"`, case-insensitive) — the
  raw labels from slices/rocking/profiles already read `"… (a.u.)"` — or when the
  user set an explicit `colorbar_label` override (used verbatim).

`_tick_formatter` is extended so `"arb"` is recognised (returns a sentinel or is
handled directly in `add_colorbar`); its existing `auto`/digit/negative-guard
behaviour is preserved.

## Component 3 — threading the group to every call site

The group is already known wherever a colourbar is built; pass it through.

- **`render.layer_figure(...)`** gains a `group=None` param and forwards it to
  `add_colorbar`. Fed by **`figures.volume_layer_specs`**, which already takes
  `cmap_group` — pass that same value as `group`. This one edit covers
  **mosaicity**, **rocking**, and **visualize** (all render via `layer_figure`).
- **`strain.py`** — both `add_colorbar` calls pass `group="strain"`.
- **`slices.py`** — pass `group=_GROUP_BY_KIND[kind]` (already computed for the
  colormap; store the group in `prep` alongside `cmap_name`).
- **`profiles.py`** — the reference-image colourbars pass the group derived from
  the field's `kind` (map kind→group; reuse/replicate `_GROUP_BY_KIND`). Confirm
  during implementation that profiles' kinds align with the four groups.

No other module calls `add_colorbar` (verified by grep: only `render.py`,
`strain.py`, `slices.py`, `profiles.py`).

## Component 4 — GUI (`gui/widgets/export_dialog.py`)

Replace the single "Tick format" row with a **"Colourbar — per group"**
subsection: one compact row per group (4 rows, not 12). Each row's label is the
group's friendly name (Mosa misorientation / Mosa FWHM / Strain / Raw intensity)
and its field is a single horizontal layout of three widgets:

```
[ format ▾ ]  [ offset size ]  [ position ▾ ]
```

- format dropdown: the `_TICK_FMTS` list; `"arb"` label reads
  "arbitrary units (no ticks)"; `"scientific"` label unchanged.
- offset size: `QDoubleSpinBox`, e.g. range 0.2–5.0, step 0.1, binds
  `offset_scale_<group>`.
- position dropdown: `["top", "bottom"]`, binds `offset_pos_<group>`.

The global colourbar controls (show, label override, fraction, tick count,
round-limits) stay as they are. `_TICK_FMTS`, `_TICK_FMT_LABELS`,
`sync_from_style`, `_all_widgets`, and `_build_controls` are updated for the new
widgets.

**Targeted improvement (in scope because we are expanding the control count):**
wrap `StyleControls` in a `QScrollArea` inside `ExportDialog` (and the
publication-style dialog in `main_window._on_pub_style`, which reuses
`StyleControls`) so the taller form never overflows the dialog.

## Component 5 — retire the title-pad hack (follow-through)

With scientific offset text movable to the bottom, the top-collision that
`_TITLE_PAD_PER_FONT_SCALE` compensates for no longer occurs when offset is at the
bottom. Do **not** remove the hack outright (offset can still be placed at the
top): leave it as-is. Note it here so the implementer knows why it exists and does
not "fix" it. No change required.

## Error handling

- `tickfmt_for`/`offset_scale_for`/`offset_pos_for` tolerate `group=None`; unknown
  non-None groups raise `KeyError` (consistent with `cmap_for`).
- Hand-written / stale persisted styles with an out-of-range tick-format value
  fall through to `auto` (as `_tick_formatter` already does for unparseable input).
- `offset_scale` is floored (e.g. `max(value, 0.1)`) before use so a 0 from a
  stale style cannot produce a zero-size font, mirroring the scale-bar label
  guard.
- Degenerate colour limits (`maxabs == 0`) give `oom = 0` and suppress the
  exponent label rather than dividing by zero.

## Testing

- Update the tests that reference the removed `colorbar_tick_format` field to the
  per-group fields: `tests/test_plot_style.py` (`add_colorbar` call, the
  scientific/`colorbar_ticks` cases, the constrained-layout offset case) and
  `tests/test_figure_layout.py` (`_BIG` style).
- New unit tests:
  - `tickfmt_for` / `offset_scale_for` / `offset_pos_for` including `group=None`
    and unknown-group `KeyError`.
  - `_tick_formatter("arb")` behaviour and the arbitrary-units path (no ticks +
    " (arb. units)" label; label override wins).
  - Scientific renderer: correct `oom` from limits, mantissa tick formatting,
    built-in offset hidden, one custom exponent `Text` present at the requested
    top/bottom position with size scaling by `font_scale * offset_scale`.
  - `_style_from_dict` migration: a dict lacking all `tickfmt_*` keys adopts the
    tuned profile; a dict with them is left untouched.
- Regression guard: assert the `style=None` `layer_figure` colourbar is unchanged
  (bare `PlotStyle()` stays `auto`).
- `tests/gui_smoke.py`: update if it counts `StyleControls` widgets.

## Documentation (same change)

- `docs/Usage.md` — Colourbar section: per-group tick format, the arbitrary-units
  option, and the offset size/position controls.
- `docs/Codebase.md` — `PlotStyle` field list, the three `*_for` helpers, and the
  new `add_colorbar(..., *, group=None)` signature + scientific/arbitrary
  behaviour.

## Files touched (summary)

- `dfxm/common/plotting.py` — schema, helpers, `PUBLICATION_STYLE`, migration,
  `_tick_formatter`, `add_colorbar`.
- `dfxm/common/render.py` — `layer_figure(group=…)`.
- `dfxm/common/figures.py` — `volume_layer_specs` forwards `cmap_group` as `group`.
- `dfxm/stages/strain.py`, `slices.py`, `profiles.py` — pass `group=` at each
  `add_colorbar` call.
- `gui/widgets/export_dialog.py` — per-group controls + `QScrollArea`.
- `gui/main_window.py` — `QScrollArea` around the reused `StyleControls` (if the
  publication-style dialog needs it).
- `tests/…`, `docs/Usage.md`, `docs/Codebase.md`.

## Out of scope / YAGNI

- Per-group `colorbar_ticks` count, per-group colourbar fraction, or per-group
  label overrides — global remains fine.
- Removing the `_TITLE_PAD_PER_FONT_SCALE` hack (top placement still uses it).
- Any change to histogram / 3-D / paraview rendering.
