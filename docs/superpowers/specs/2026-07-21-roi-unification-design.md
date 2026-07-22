# ROI unification — design

**Date:** 2026-07-21 (brainstormed 2026-07-19–21)
**Status:** approved by Albert (section-by-section)
**Motivation:** the 2026-07-18 incident — darfix's origin+size numbers typed into
rocking's start,end fields produced a raw-mosa volume misregistered ~154 µm
along Y. ROI entry today is a mishmash of frames and conventions spread over
five stages with no single source of truth.

## Problem

One physical window is expressed in at least three conventions across the app:

| Where | Field(s) | Frame today |
|---|---|---|
| darfix (external) | ROI widget | origin + size |
| rocking | `roi_x`, `roi_y` | absolute detector start,end |
| visualize / paraview | `roi_x`, `roi_y` | map-frame (help text wrongly says "detector") |
| slices | `align_roi_x/y` | map-frame; applies only to stacked volumes |
| strain | `roi` (`r0,r1,c0,c1`) | map-frame, cropped after detrend |
| paraview | `mosa/strain_darfix_origin_xy` | darfix origin, STO2 value hard-coded as default |
| experiment YAML | — | nothing stored |

Two distinct concepts hide under "ROI":

1. **Darfix window** — origin+size of the darfix detector crop. A *fact* about
   how the maps were made; registration-critical.
2. **Analysis window** — the sub-region chosen for study (STO2: map-frame
   y 400,1100 = detector rows 630,1330). A *choice*.

Users must convert between frames by hand; errors co-register silently.

## Goals (agreed)

- **Error-proof + type once:** one canonical entry; every stage derives its own
  frame automatically. (Not just guards, not just relabeling.)
- Both ROIs live on the **experiment** (preset YAML), edited in the
  Edit-experiment dialog.
- Entry convention **matches each source**: darfix window as origin+size
  (verbatim copy from darfix), analysis window in map-frame start,end with the
  derived detector rows always displayed.
- Stages **pre-fill + override**: fields stay editable (advanced use), with a
  visible deviation marker when a value differs from the experiment.
- In scope: frame-honest labels/help sweep; ROI picker in the experiment editor.
- Out of scope: replot-dialog crop pre-fill (presentation-only);
  paraview `mosa/strain_darfix_origin_xy` params are **completely untouched**
  (no retirement, no pre-fill — Albert's explicit call).

## Approach

Thin core derivation layer (approved "Approach 1 + frame metadata"): plain
string fields on `Experiment`, pure conversion functions in a new Qt-free
module, pre-fill wired through the existing `experiment_overrides` mechanism,
plus a cheap test-enforced `Param.roi_frame` declaration (no generic conversion
machinery).

## Section 1 — Data model & core (Qt-free)

`Experiment` gains three string fields (blank = no crop):

- `darfix_roi: str = ""` — `"x,y,w,h"` origin+size, verbatim from the darfix
  ROI widget (STO2: `"105,230,1832,1266"`).
- `analysis_roi_x: str = ""` — map-frame `"c0,c1"` columns.
- `analysis_roi_y: str = ""` — map-frame `"r0,r1"` rows (STO2: `"400,1100"`).

Split x/y (not one 4-tuple) to mirror the per-axis stage params and the
picker's axis split. All three join `EXPERIMENT_SCHEMA` in a new
"Regions of interest" group; the STO2 preset gets the real values.

New module **`dfxm/common/roi.py`** — pure functions, no Qt, no I/O:

- `parse_darfix_roi(s)` → `DarfixWindow` dataclass (origin/size; derived
  detector `x0,x1,y0,y1`).
- `map_to_detector(pair, origin)` / `detector_to_map(pair, origin)` — the one
  conversion, written once.
- `analysis_detector_window(exp)` → absolute detector start,end pairs (what
  rocking needs).
- `validate_rois(exp)` → list of human-readable problems (malformed pairs,
  end ≤ start, analysis outside the darfix window).

`Param` gains **`roi_frame: str = ""`** — `"" | "detector" | "map"`, validated
in `__post_init__` like `roi_axis`. Documentation-only metadata:
`tests/test_param_metadata.py` enforces that every ROI param (any with
`roi_group`, plus rocking's `roi_x`/`roi_y`) declares its frame and mentions it
in `help`. No conversion machinery hangs off it.

## Section 2 — Experiment editor UI

The Edit-experiment dialog gains a **"Regions of interest"** section:

- **Darfix ROI (origin + size)** — one row of four inputs `x, y, w, h`; help:
  "copy these four numbers straight off darfix's ROI widget — no conversion".
- **Analysis window X / Y (map px)** — two `start,end` fields, map-frame.
- **Derived read-out** — read-only label, updated live as fields change:
  "Detector window: x 105→1937, y 230→1496 · analysis in detector rows:
  y 630→1330". Both frames always visible; the previously hand-derived numbers
  are shown, never typed.
- **Pick… button** — opens the existing matplotlib ROI picker on a mid-stack
  layer of a `maps.h5` (auto-suggested from `processed_root` +
  `maps_filename`; Browse fallback). The picker works natively in the map
  frame, so the dragged box fills `analysis_roi_x/y` directly. Lazy import;
  nothing built until clicked (house rule).
  Implementation deviation (Task 6): the preview source is the *stacked
  volumes* beside `processed_root` (`stacked_volumes.h5` /
  `stacked_strain_volumes.h5`, via the existing
  `dfxm.common.figures.stacked_volume_previews`) rather than a per-layer
  `maps.h5` — same map frame, and it reuses that helper unchanged instead of
  teaching the dialog a new HDF5 layout.
- **Validation on OK/Save** — `validate_rois` runs; problems shown inline with
  a hint (StageUserError-banner pattern). Malformed ROIs block saving; blank
  ROIs are fine.

The darfix ROI stays typed-only (picking it would need a raw detector frame —
out of scope).

## Section 3 — Stage pre-fill & override

`gui/bindings.py` `experiment_overrides` derives each stage's fields in its
**native frame** via `dfxm/common/roi.py`:

| Stage | Field(s) | Pre-filled with |
|---|---|---|
| rocking | `roi_x`, `roi_y` | analysis window in absolute detector px (darfix origin + map values → STO2 `105,1937` / `630,1330`) |
| visualize | `roi_x`, `roi_y` | analysis window, map-frame, as-is |
| paraview | `roi_x`, `roi_y` | analysis window, map-frame, as-is |
| slices | `align_roi_x`, `align_roi_y` | analysis window, map-frame, as-is |
| strain | `roi` | `r0,r1,c0,c1` assembled from analysis y + x |

Blank experiment ROIs pre-fill nothing — stages behave exactly as today.
`mosa/strain_darfix_origin_xy` on paraview: untouched.

**Override behaviour** follows the existing architecture: fields stay editable;
pre-fill rides the same `experiment_overrides` path as calibration/chained
outputs; form-state persistence keeps working (a genuinely-edited override is
restored per experiment, a never-touched field follows the experiment).

**Deviation marker (new):** when a ROI field's current value differs from the
experiment-derived value, its label gets a visible hint (e.g. ⚠ suffix with
tooltip "differs from experiment: 630,1330"). Stale persisted overrides stop
being invisible.

## Section 4 — Validation, labels, docs, tests

**Validation & guards.** `validate_rois` errors surface only in the editor at
save. Stages keep their current `StageUserError` paths. The slices
`_y_height_notes` guard stays as the last line of defence (catches hand-edited
overrides too).

**Frame-honest labels/help sweep** — every ROI param states its frame:

- visualize/paraview `roi_x/roi_y`: reword help (currently claims "detector")
  to map-frame — "columns of the darfix map, 'c0,c1' px"; labels become
  "Map ROI X/Y".
- slices `align_roi_*`: same map-frame wording.
- strain `roi`: states map-frame + row/col order.
- rocking `roi_x/roi_y`: keeps the darfix origin+size warning; gains
  "pre-filled from the experiment ROIs — normally leave as-is".

**Docs (same change, per contract).** `Usage.md`: new "ROI frames" subsection
(darfix window = fact, analysis window = choice, one worked STO2 example), the
experiment-editor ROI section, updated per-stage ROI paragraphs. `Codebase.md`:
`dfxm/common/roi.py`, the three `Experiment` fields, `Param.roi_frame`, the
bindings derivations.

**Tests.**

- `roi.py` unit tests: round-trips + STO2 golden case (origin 105,230 size
  1832,1266; analysis y 400,1100 → detector 630,1330 — the incident's
  hand-conversion, now asserted as a derivation); validation rejects malformed
  pairs, end ≤ start, and analysis windows outside the darfix size. (The
  incident itself is closed structurally: rocking's detector window is derived,
  never typed.)
- `test_param_metadata.py`: every ROI param declares `roi_frame` and mentions
  the frame in help.
- Bindings tests: per-stage pre-fill from a fixture experiment, incl. rocking's
  derived detector window and the blank-ROI no-op.
- Editor: validation + derived read-out unit tests; a `gui_smoke` step opening
  the ROI section.

## Ties to existing wishes

- Subsumes the darfix-ROI half of the experiment-initializer wish
  (`experiment-initializer-wish` memory); the positioner/pixel-size wizard half
  remains future work.
- Advances the tooltip-precision pass for ROI params specifically.
