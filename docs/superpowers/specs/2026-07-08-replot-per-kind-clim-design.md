# Replot: per-kind colour limits + h5-relative output default

**Date:** 2026-07-08
**Scope:** the two replot dialogs (`SliceReplotDialog`, generic `ReplotDialog`)
and the four `render_replot` cores (slices, strain, mosaicity, rocking).

## Problem

The Replot dialogs expose a single `vmin/vmax` pair applied to *every* selected
plane/layer regardless of what it is. When a file mixes several plot kinds — the
slices `oblique_slices.h5` holds `mosa_com`, `mosa_fwhm`, `strain`, `raw`; a
mosaicity stacked h5 holds `mosa_com` + `mosa_fwhm` — one global colour limit is
meaningless. Users want to set the colour limits per kind.

Separately, the output-dir default is pre-computed once in `stage_view.py` from
the stage form. On a cold start (Browse/Load a different h5) or when the form's
`output_dir` was unset, that default no longer points beside the file being
replotted (it can land in the CWD). The output should default to a subfolder of
the *loaded h5's own folder* and follow the file.

## Design

### Core — `resolve_clim` (new, `dfxm/common/figures.py`)

```python
def resolve_clim(clim, key):
    if clim is None:
        return None
    if isinstance(clim, dict):
        return clim.get(key)   # None -> stored/default for that kind
    return clim                # legacy single tuple -> applies to all
```

`render_replot`'s `clim` parameter widens to `None | (vmin, vmax) | {key: (vmin, vmax)}`.
Each of the four `render_replot`s calls `resolve_clim` inside its existing
per-selection loop and passes the resolved single tuple to the unchanged
`_rebuild_*` / `render_volume_layer` helper:

- **slices** — key is the derived kind-group. `ReplotEntry` gains a `group: str`
  field (`GROUP_BY_KIND[kind]`), filled by `replot_catalog`; render looks up
  `resolve_clim(clim, entry.group)`.
- **strain / mosaicity / rocking** — key is the `ReplotGroup.key`; render looks
  up `resolve_clim(clim, key)`.

Fully backward-compatible: `None` and bare-tuple `clim` behave exactly as today,
so every existing `render_replot` test is untouched.

### GUI — `ClimGroupSection` (new, `gui/widgets/clim_section.py`)

A `QWidget` that, given a list of `(key, label)` groups, builds one `vmin/vmax`
row per group, preserves already-typed values across a rebuild, validates the
boxes, and yields `clim_by_group() -> {key: (vmin, vmax)} | None`. Both dialogs
embed it and rebuild it in `_reload()` from the loaded catalog:

- generic `ReplotDialog` — rows keyed by `ReplotGroup.key`, labelled `grp.label`.
- `SliceReplotDialog` — rows keyed by the kind-groups present (`mosa_com`,
  `mosa_fwhm`, `strain`, `raw`) in a fixed order, with friendly labels.

`render_selection` passes `clim_by_group()` as `clim`; `_on_render` validates
via the section first.

### GUI — output-dir auto-default (both dialogs)

Each dialog owns a timestamp `self._ts` and defaults the output field to
`{dirname(abspath(loaded h5))}/replots/{ts}`. `_reload()` re-derives it on
Browse/Load **unless** the user has manually edited the field (tracked via
`QLineEdit.textEdited` and the Browse-output button) or an explicit non-empty
`out_default` was passed (pins it). `stage_view.py` stops pre-computing the path
for both dialogs (passes `out_default=""`); an empty loaded-h5 path yields an
empty default (fills in when a file is chosen).

## Non-goals

- Not merging the two dialog classes (different tree depth; out of scope).
- Not changing the render/appearance of any figure beyond the clim override that
  already existed.

## Tests

- `resolve_clim` unit test (None / tuple / dict / missing-key).
- `slices.replot_catalog` exposes `group`; `render_replot` per-group dict clim.
- `mosaicity.render_replot` with a 2-group dict clim (distinct limits per group).
- `strain` / `rocking` dict-clim smoke (single group).
- `ClimGroupSection` widget test (build, value-preserve, collect dict, validate).
- Update `test_gui_replot_dialog.py` (single `_vmin` → per-group section; clim
  assertion tuple → dict) and add a 2-group collection assertion.
- `SliceReplotDialog` output-dir auto-defaults to `{h5 dir}/replots/...` when
  `out_default` is empty.

## Docs (same change)

- `docs/Usage.md` — both Replot sections: per-kind colour limits + output-dir
  default.
- `docs/Codebase.md` — `resolve_clim`, `ReplotEntry.group`, `ClimGroupSection`,
  both dialogs, the four `render_replot` signatures.
