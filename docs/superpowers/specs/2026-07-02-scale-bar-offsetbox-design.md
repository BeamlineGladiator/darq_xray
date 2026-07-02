# Scale-bar offsetbox rebuild — design

**Date:** 2026-07-02
**Status:** approved

## Problem

`draw_scale_bar()` (`dfxm/common/plotting.py`) hand-computes the background
box in data coordinates with a *fixed* text allowance (`0.06 × y-extent`) and
a box width equal to the bar length. Consequences, visible in today's styled
test renders:

1. The box does not grow with the label font — at larger `font_scale` /
   `scale_bar_label_scale` the text spills above and beside the box.
2. When the label is wider than the bar, the text pokes out both sides of the
   box; the bar/text group is not presented as one centred unit.

A per-label scaling knob already exists (`scale_bar_label_scale`, GUI "Label
scale"), and bar thickness is independently controllable — so **no new scale
factor is added**; the fix is geometry only.

## Decision

Rebuild the body of `draw_scale_bar()` on matplotlib's offsetbox machinery
(the canonical "anchored size bar" construct). Signature and call sites are
unchanged (`dfxm/stages/strain.py`, `dfxm/stages/slices.py`,
`dfxm/common/render.py`).

Rejected alternative: keep the hand-rolled patches and measure the rendered
text extent. Smaller diff, but points→data conversion at call time is
systematically off under constrained layout (axes position is finalised at
first draw) — the same class of approximation that caused the bug. Offsetbox
layout happens at draw time, so it is exact by construction.

## Construction

```
AnchoredOffsetbox(
    loc=style.scale_bar_loc,          # "lower left" … strings pass through
    child=VPacker(align="center", sep=<gap>, children=[
        TextArea(label, textprops=...),          # label on top
        AuxTransformBox(ax.transData) ← Rectangle((0,0), length_um, bh),
    ]),
    frameon=style.scale_bar_box,
    pad=<margin_pt in font-size units>,
    borderpad=<distance from axes edge>,
)
```

added via `ax.add_artist`, high zorder.

Style-knob mapping (all existing controls keep working, no schema change):

| PlotStyle field            | Maps to                                            |
| -------------------------- | -------------------------------------------------- |
| `scale_bar_length_um`      | bar Rectangle width (data µm via AuxTransformBox); `None` → `auto_scale_bar_length_um` unchanged |
| `scale_bar_thickness_pt`   | bar height, **unchanged formula** `0.004 × pt × |y-extent|` |
| `scale_bar_label_scale`    | label fontsize `10 × font_scale × label_scale`, bold |
| `scale_bar_color`          | bar facecolor + label colour                       |
| `scale_bar_loc`            | `AnchoredOffsetbox(loc=...)` (same strings)        |
| `scale_bar_box`            | `frameon`                                          |
| `scale_bar_box_color` / `_alpha` | frame `patch` facecolor / alpha, `edgecolor="none"`, rounded boxstyle kept |
| `scale_bar_box_margin_pt`  | frame padding **in real points** (converted to the offsetbox's font-size units) |

Text–bar gap (`sep`): `0.25 × label fontsize` in points. Anchor inset from
the axes edge (`borderpad`): `0.5` font-size units (matplotlib's convention).

## Behaviour changes (accepted)

- Box always hugs the rendered text + bar at any font scale; text and bar are
  mutually centred by the `VPacker` (bar centred beneath a wider label).
- Box padding/anchor inset are constant in points, no longer proportional to
  the data extent ("Box margin (pt)" finally means points).
- Both render paths change (styled and `style=None`); both are already
  documented as not byte-identical to the pre-export legacy renderer
  (`render.py:41-45`), and the only pinned legacy property is the absent
  layout engine, which this does not touch.
- The bar/label artists move from `ax.patches`/`ax.texts` into an
  `AnchoredOffsetbox` in `ax.artists`.

## Tests (`tests/test_plot_style.py`)

- Rewrite the four existing scale-bar tests to query the `AnchoredOffsetbox`
  in `ax.artists` (presence, "µm" label, frame on/off, margin → padding).
- New regression for the reported bug: render with `FigureCanvasAgg` at
  `font_scale=2.2`, assert the label's window extent is contained within the
  frame patch extent, and the label centre x ≈ bar centre x.

## Docs (same change)

- `docs/Usage.md`: scale-bar box auto-sizes to the label; margin is in points.
- `docs/Codebase.md`: updated `draw_scale_bar` description.
