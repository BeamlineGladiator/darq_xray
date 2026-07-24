# Figure-builder node inspector + hardening sweep — design

**Date:** 2026-07-24
**Status:** approved (brainstormed with Albert; follow-up to the merged figure-builder
project, master 0c87e76 — realizes the fable final review's designated follow-up)

## Problem

The merged figure builder's recipe model and renderer fully support `Col.shared_x`,
group `shared_colorbar`/`shared_clim`, `Row.pinned_height_cm`/`Col.pinned_width_cm`,
and `Spacer`/`TextCell` box sizes — but the GUI has no widgets for any of them, so
the spec's acceptance figure 2 (shared-x trace stacks) is only authorable by
hand-editing recipe JSON. The outline also drops selection after every mutation
(move up/down cannot be pressed twice in a row). Separately, the whole-branch review
triaged a minor backlog (correctness, consistency, cosmetics) as follow-up work.

## Scope (agreed)

1. Node-properties authoring in the GUI (all node types).
2. Outline selection persistence.
3. The full minor-hardening backlog — all three tiers.

## Design

### A. Node inspector (gui/figure_builder.py, right pane)

Replace the panel-only "Selected node" box with a `QStackedWidget` keyed by the
selected node's type:

- **Panel** page: the existing override editor, unchanged.
- **Row** page: group label edit, pinned height cm spinbox (0 = off → None),
  shared-colorbar controls.
- **Col** page: as Row, plus `shared_x` checkbox and pinned width cm (0 = off).
- **Spacer** page: width/height cm spinboxes.
- **TextCell** page: text edit + box width/height cm spinboxes.
- No selection → hint label.

Shared-colorbar controls = "One colorbar for this group" checkbox writing the
node's `shared_colorbar` field + optional clim override text ("lo,hi", blank =
union of member ranges) writing `shared_clim`. Member panels' own `colorbar`
overrides are not touched (the renderer already suppresses member bars).

Editing discipline (established by the merged project, binding here):
- Per-key submission — an edit submits only the touched field; no cross-field
  clobber (the 3-state label lesson).
- Parse failures (clim text, etc.) → notes-bar message, no mutation, no crash.
- Every real mutation marks dirty and schedules the debounced preview; no-op
  edits must not dirty.

The label dialog's leaked "auto" sentinel (shown as prefill for auto-grouped
nodes) is fixed in the same region: display blank, keep the sentinel internal.
The inspector's label controls expose all three panel-label states (auto /
suppressed / manual text) — suppression via an explicit control, not magic text.

### B. Outline selection persistence (gui/figure_builder.py)

`_rebuild_tree` captures the selected recipe node (object identity) before
teardown and re-selects the rebuilt item whose stored `UserRole` node `is` that
object. `delete_selected` selects the deleted node's parent container
afterwards. Result: move up/down keeps the moved node selected (repeatable);
group toggles and label edits keep context.

### C. Acceptance

A GUI-side test builds the merged spec's **figure 2** end-to-end through window
methods only (no JSON editing): ragged 3-column layout, shared_x stacks with
group labels, shared scale-bar treatment, split map/trace scales — and asserts
the rendered geometry matches the existing acceptance expectations (reuse the
fixture + box arithmetic from tests/test_compose_acceptance.py). This is the
proof the authoring gap is closed.

### D. Hardening sweep — correctness/robustness

Each lands where the code lives, failing-first test per item:

1. `recipe.py`: malformed v1 recipes (unknown compose keys → TypeError; missing
   panel `id`/`source` → KeyError) wrapped into `StageUserError("recipe is
   malformed", hint)`.
2. `validate_recipe`: duplicate `PanelRef`s to one panel id refused (today the
   first axes is silently left empty).
3. `render.py`: `scale_bar_mode="one-panel"` with a non-map or unplaced target →
   `StageUserError` (today: silently no bar anywhere).
4. CLI: unwritable/uncreatable `-o` directory → exit 2 message+hint (today: raw
   OSError traceback from `os.makedirs`).
5. `layout.py` nested pins: (a) trace inside both Row(pinned_height) and
   Col(pinned_width) — Row pin currently silently dropped; (b) pinned-height Row
   propagating the full pin through a nested Col so the column overflows.
   Fix the propagation math — a pinned dimension shared by stacked children is
   divided among them after subtracting inter-panel gutters, so the container's
   total equals the pin — and emit implied-scale notes; never silent.
6. Zero-length traces (`length_um == 0`) join the degenerate-extent placeholder
   lockstep (placeholder kind + note, not a 0-width trace cell).
7. Loaded recipes with orphaned `PanelDef`s (no layout ref): skipped from
   loading entirely (no wasted h5 read; render already tolerates them).

### E. Hardening sweep — consistency/UX

8. `group_label` semantics: blank/"" group label = "no label" consistently
   (vs panel label "" = suppressed); document the difference where the two
   appear (`recipe.py` docstrings + Codebase.md).
9. Stale preview: when the last panel is deleted, `render_now` clears the
   canvas instead of leaving the previous figure behind the "add panels to
   preview" note.
10. `export_recipe` honors `style_overrides` for formats/dpi (today it reads
    `recipe.style` only — latent asymmetry).

### F. Hardening sweep — cosmetics/DRY

11. `adapters.py`: extract shared `_crop_uv(plane, u, v, roi)` used by the three
    UV-cropping loaders.
12. `recipe.py`: non-recipe JSON (no version/layout/panels keys) gets a "not a
    recipe file" message instead of "unsupported recipe version None".
13. CLI `--formats` multi-bad-value message quotes values individually.
14. Codebase.md: split the `_draw_reference_image` run-on entry.
15. `layout.py`: add the brief-documented type hints to
    `measure_cells`/`place_tree`.

### G. Error handling & testing

- All new refusals follow `StageUserError(message, hint)`; GUI paths surface via
  the notes bar (never crash); nothing silent — notes for every degradation.
- Failing-first tests per behaviour change; the figure-2 GUI acceptance test
  (§C) is the headline gate; suite baseline 735 passed / 13 skipped /
  0 warnings must hold throughout; gui_smoke extended only if a new user flow
  needs it (inspector interactions belong in pytest GUI tests).
- Docs same-change contract: Usage.md (inspector workflow — how to set shared
  axes/bars/pins/sizes; selection behaviour) + Codebase.md (inspector methods,
  changed core functions) in the same commits.

## Out of scope

- New recipe-model capabilities (histogram/matched panels, visualize catalog,
  journal presets) — unchanged from the merged project's deferred list.
- Real-data eyeball / PDF-SVG inspection — separate validation session.
