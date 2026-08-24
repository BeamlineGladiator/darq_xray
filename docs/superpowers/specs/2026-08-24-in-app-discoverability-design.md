# In-app discoverability pass — design

Date: 2026-08-24
Status: approved in brainstorming; awaiting user review before planning
Baseline: master `7dc3c91`

## Problem

The pipeline flow itself tests as approachable. Friction is concentrated in one
shape: **a feature lives somewhere other than where intuition says it lives.**
A first-time user looking for colormaps opens the stage form's *Appearance*
group and finds vmin/vmax but no colormap — because colormaps are set per
quantity group in the global *Publication style…* dialog, reachable only from a
bare button in the left column.

Five friction points were recorded in the 2026-08-07 user-perspective
walkthrough. A sweep of the current tree (this session) confirmed four of them,
sized the fifth down, and found three more.

### What the sweep found

- **Parameters are already covered.** All 187 params across the nine stages
  have `help` text (the 2026-07-26 tooltip sweep). The uncovered surface is
  *widgets*: 36 `setToolTip` calls app-wide, and `gui/main_window.py` has
  exactly two (the darfix note and the theme toggle). `Publication style…`
  and `Figure builder…` are bare buttons with no tooltip at all.
- **`mosaicity` and `concat` have no `Appearance` group** — and mosaicity has
  no plotting parameter of any kind. It nonetheless produces styled maps whose
  colormaps come from the global dialog. Any pointer mechanism must work with
  no field and no group to anchor to. This is the single fact that decides the
  mechanism in §1.
- **The 3-D tab is less inert than reported.** `gui/widgets/volume3d.py:39`
  already shows *"(run the stage, then open a volume in the 3-D viewer)"*. What
  remains is that the tab *label* offers no cue before you click it.
- **`Export…` / `Export all…` start disabled** (`gui/stage_view.py:237`) and
  enable only after a run completes (`:920`), with nothing saying so.
- **`matched.colormap` is genuinely the only outlier.** Per-stage `vmin`/`vmax`
  and percentile clips are normal (strain, rocking and matched all carry them);
  only *colormap* breaks the per-group rule. `matched` never passes a
  `cmap_group` at all — it is the "matched.py not group-wired" item deferred
  from the per-group-colorbar project.
- **`jobs_json` is a non-advanced `TEXT` param in the profiles form's
  ungrouped top section** — a wall of raw JSON is the first thing the stage
  shows, despite `Pick line…` and `Jobs from marks…` existing to fill it.

### Decisions taken during brainstorming

1. **Scope:** the five walkthrough points *plus* the sweep findings above.
2. **Audience:** anyone, always-visible and quiet. **No first-run state, no
   dismissal persistence, no tour.** Nothing that ages out or needs clearing.
3. **Mechanism:** schema-declared, rendered **both** inline under the field and
   in the help panel.
4. **`matched`:** group-wire it and **keep** the `colormap` param as the
   no-group fallback (not a breaking removal).
5. **`jobs_json`:** a summary line plus an `Edit raw JSON…` dialog.
6. **Style-at-Run:** stamp the run and note it in the style dialog — *not* a
   third line next to the Run button.

## Non-goals

- No first-run overlay, guided tour, or dismissible hint layer (decision 2).
- No removal of `matched.colormap`, and no change to any other stage parameter.
- No general refactor of `ParamForm`, `HelpPanel` or `StageView` beyond what
  these items need.
- `concat` and `paraview` receive no colormap pointer: concat produces no
  figures, and paraview writes VTI whose colormap is chosen inside ParaView.

## §1 — Mechanism

One new carrier on the stage schema, rather than five ad-hoc widgets:

```python
@dataclass(frozen=True)
class SeeAlso:
    """A pointer from where a user looks to where the feature actually lives."""
    anchor: str   # "" = stage-level | "group:Appearance" | "param:colormap"
    text: str     # one terse sentence

@dataclass(frozen=True)
class StageSpec:
    ...
    see_also: tuple[SeeAlso, ...] = ()
```

**Why on `StageSpec` and not mirroring `advice_key` on `Param`:** `mosaicity`
has neither an `Appearance` group nor any plotting parameter, yet needs the
colormap pointer. A field-only carrier has nowhere to put it. One anchored list
covers field-, group- and stage-level cases with a single rendering pass and a
single docs entry.

`SeeAlso` lives in `dfxm/config/models.py` beside `Param`; it is plain data and
keeps `dfxm/` Qt-free.

This is the carrier for *pointers*. Item 2.4 adds one further, unrelated schema
field (`Param.editor`, a render hint) — the two are independent additions and
neither subsumes the other.

### Rendering — inline (A)

`ParamForm` grows one rendering pass alongside the existing `_add_note_row`
(`gui/widgets/param_form.py:134`). Each `SeeAlso` becomes a word-wrapped,
**always-visible** `QLabel` with `setProperty("role", "hint")` — deliberately
distinct from the `role="warning"` advisory notes, which are hidden by default
and carry cost warnings. Placement by anchor:

| anchor | placed |
|---|---|
| `""` | at the top of the form, above the first section |
| `"group:X"` | directly under group `X`'s header label (`param_form.py:107`) |
| `"param:p"` | directly under `p`'s editor row |

The editor widget itself is **not** wrapped — `self._editors[name]` must stay
the real widget, which `gui_smoke` and the wheel-guard tests reach into
directly. This is the same constraint `_add_note_row` already documents.

### Rendering — help panel (C)

- `param_help_html` (`gui/widgets/help_panel.py:20`) appends a `See also:` line
  when a `param:` anchor matches the focused param.
- `HelpPanel.set_idle` (`:66`) appends stage-level (`anchor=""`) entries to the
  stage description, so they are visible whenever no field has focus — which is
  how every stage opens (`stage_view.py:292`).

The same sentence therefore appears inline and in the help panel. That
redundancy is the deliberate cost of choosing A + C: inline makes it findable
without clicking, the help panel makes it readable where users already look for
explanations.

### Anchor validity

A typo'd anchor must fail the suite, not silently render nothing. A Qt-free
enforcement test walks every `StageSpec` in `gui/bindings.STAGE_SPECS` and
asserts each anchor is `""`, names a group that at least one param declares, or
names a param that exists. This mirrors the existing spec-enforcement tests.

## §2 — The surfaces

### 2.1 Colormap pointers

`SeeAlso` entries only; no behaviour change.

| stage | anchor |
|---|---|
| strain, rocking, visualize, profiles | `group:Appearance` |
| slices | `group:Quantities` |
| matched | `param:colormap` |
| mosaicity | `""` (stage-level) |
| concat, paraview | none |

Wording is per-stage but built from one sentence: colormaps are set per
quantity group in *Publication style…* in the left panel; the vmin/vmax and
percentile fields here are this stage's own. The matched entry instead says
that Publication style drives standard quantities and this dropdown is the
fallback for anything without a quantity group (see 2.2).

### 2.2 `matched` group-wiring

`dfxm/stages/matched.py` currently passes `p["colormap"]` straight through
(`:588`, `:615`, `:650`, `:669`) and never supplies a `cmap_group`. It changes
to resolve the quantity group like every other stage and call
`resolve_cmap(style, group, fallback=p["colormap"])`
(`dfxm/common/figures.py:274` is the existing pattern), using the shared
`GROUP_BY_KIND` map (`dfxm/common/plotting.py:54`).

**This is a real behaviour change: matched figures for standard quantities will
follow the publication style on the next run instead of the per-stage
dropdown.** The `colormap` param is kept, so no saved form state, preset or CLI
invocation breaks; it becomes the fallback for quantities with no group. Both
docs are updated in the same commit.

### 2.3 Style-at-Run signalling

Two surfaces, chosen so each addresses the point where the misunderstanding
actually occurs, and deliberately **not** a third line beside the Run button
(that spot already carries the cost readout and the pre-flight banner).

- **Stamp the run.** `StageView._on_finished` (`gui/stage_view.py:890`) appends
  a line recording the style the run actually used: exactly the four
  quantity-group colormaps (`cmap_mosa_com`, `cmap_mosa_fwhm`, `cmap_strain`,
  `cmap_raw`) and the font scale. Appended once in the common path, **not** in
  each `_summarize_*` function.
- **Note in the dialog.** The Publication style dialog
  (`gui/main_window.py:233-241`) states that edits apply to runs started from
  now on, and points at `Replot…` to restyle finished runs.

### 2.4 `jobs_json` summary editor

A new `gui/widgets/jobs_summary.py` renders a read-only summary label plus an
`Edit raw JSON…` button that opens the raw text in a dialog.

Selected by a **generic** render hint on the schema: `Param.editor: str = ""`,
with `editor="summary_json"` meaning *a TEXT param holding a JSON list, shown
as a summary with a raw editor behind a button*. The widget formats any JSON
list generically — it uses `name` and `offset_um` keys when present and falls
back to `"N entries"` otherwise — so **no stage-specific knowledge enters the
GUI**, consistent with the schema-driven convention.

Contract that keeps the pickers untouched:

- `values()` returns the raw JSON string, exactly as the `TEXT` editor does.
- `set_values({"jobs_json": ...})` accepts a raw string and refreshes the
  summary.

Both picker call sites (`gui/stage_view.py:544` and `:607`) therefore need no
change, and this must be asserted by test rather than assumed.

### 2.5 3-D tab cue

A tab tooltip on the `3D` tab (`gui/stage_view.py:262`). The panel's own status
line already covers the inside, so this is the tab label only.

### 2.6 Button tooltips

The sweep's widget gap. Add tooltips to:

- `Publication style…` and `Figure builder…` (`gui/main_window.py:119`, `:126`);
- every stage-view action button: `Replot…`, `Pin planes…`, `Mark planes…`,
  `Jobs from marks…`, `Pick line…`, `Pick ROI…`
  (`gui/stage_view.py:142-173`) — none currently has one;
- `Export…` / `Export all…` (`:237-241`), which additionally explain the
  disabled state ("available once a run has produced figures") and switch to
  the enabled wording at `:920`.

## §3 — Testing

Qt-free:

- anchor validity across every `StageSpec` (§1);
- `matched`'s group resolution, including the fallback path for a quantity with
  no group;
- the summary formatter, including its `"N entries"` fallback and malformed
  JSON.

Qt:

- inline note rendering for each of the three anchor kinds, and that
  `self._editors[name]` is still the real editor widget afterwards;
- help-panel `See also:` for a `param:` anchor and for the stage-level idle text;
- the `summary_json` editor's `set_values` → `values` round-trip returning the
  input string unchanged;
- tooltip presence on the buttons listed in 2.6, and the disabled/enabled
  wording switch on the export buttons.

`gui_smoke`: new entries for opening the jobs raw-JSON dialog and for the style
stamp appearing in a stage summary.

**Mutation discipline.** This project has found twenty-two checks that had
stopped checking what they name, two of them in tests a plan itself specified,
and two authored by the fix for the previous one. For every test added or
changed here: run the mutation that should break it, confirm it fails **at the
named assertion line**, and assert the precondition that keeps the fixture
inside the region the test claims to cover. Run mutation sweeps with
`python3 -B` — two same-size mutations inside one mtime second reuse a stale
`.pyc`.

## §4 — Docs contract

Both items 2.2 and 2.4 change user-visible behaviour and code structure, so
each trips the docs contract:

- `docs/Usage.md` — matched's colormap now following Publication style; the
  new jobs summary editor and its raw dialog; the style-at-Run stamp.
- `docs/Codebase.md` — `SeeAlso`, `StageSpec.see_also`, `Param.editor`, the
  `ParamForm` rendering pass, `gui/widgets/jobs_summary.py`.

Updated in the **same** commits as the code, not as a follow-up.

## Verification

Full suite with `--deselect tests/test_gui_viewer3d.py` (mandatory on this
box), `ruff check . && ruff format .`, and `DISPLAY= python3 -u
tests/gui_smoke.py` run 2–3 times — smoke step `[41]` is intermittent on an
unmodified tree and must never be attributed to this diff from a single
failure.

The pass is additive to the GUI's visual weight, so it ends with an on-screen
eyeball by the user before it is called done.
