# In-App Discoverability Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cross-cutting features findable from where a user looks for them — colormaps, style timing, the 3-D tab, the profiles job list, and the unlabelled buttons — without adding first-run state or a tour.

**Architecture:** One schema-declared carrier (`SeeAlso` on `StageSpec`) renders quiet pointer text both inline in `ParamForm` and in `HelpPanel`, so the GUI hard-codes no stage knowledge. Four independent surface fixes sit alongside it: `matched` group-wiring, a style-at-Run stamp, a summary editor for `jobs_json`, and tooltips for the buttons that have none.

**Tech Stack:** Python 3, PySide6, matplotlib (explicit `Figure` API only), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-24-in-app-discoverability-design.md` (commit `311829d`)

## Global Constraints

- **Keep `dfxm/` Qt-free.** Never import PySide6/pyvista in `dfxm/`. `SeeAlso` is plain data.
- **Don't hard-code stage fields in the GUI.** Pointer text and the editor hint live on the spec; `ParamForm` renders whatever the spec declares.
- **Docs contract:** any change to a stage's parameters, behaviour, or a viewer updates `docs/Usage.md` (user-visible) and `docs/Codebase.md` (code structure) **in the same commit**, never as a follow-up.
- **Plotting:** explicit `matplotlib.figure.Figure` API; never `pyplot`, never `matplotlib.use(...)`.
- **Read before first Edit.** Any file not created this session must be Read once before its first Edit. Never reconstruct an `old_string` from memory — `help=` strings contain em-dashes and sit at varying indentation.
- **Mutation discipline (non-negotiable in this repo).** Twenty-two checks here have been found to have stopped checking what they name; two were in tests a plan itself specified, and two were authored by the fix for the previous one. For **every** test you add or change: run the mutation that should break it and confirm it fails **at the named assertion line**, and assert the precondition that keeps the fixture inside the region the test claims to cover. Run mutation sweeps with `python3 -B` — two same-size mutations inside one mtime second reuse a stale `.pyc`.
- **Test commands:**
  - Suite: `python3 -m pytest -q --deselect tests/test_gui_viewer3d.py` (the deselect is **mandatory** — in-process Qt GL segfaults on this box).
  - Lint: `ruff check . && ruff format .` (format also runs on Write/Edit via the settings hook).
  - Smoke: `DISPLAY= python3 -u tests/gui_smoke.py` — clear `DISPLAY`; ambient `:10` causes X BadWindow in the runner child. Step `[41]` is **intermittent on an unmodified tree**; run the smoke 2–3 times before blaming a failure on your diff.
- **Baseline:** master `311829d`. Rollback for the whole branch: `git reset --hard 311829d`.
- **No git remote** — no pull/push/PR in any flow.

---

### Task 1: `SeeAlso` on the schema

Adds the Qt-free carrier and its validation. Nothing renders yet.

**Files:**
- Modify: `dfxm/config/models.py` (add `SeeAlso` near `Param`; add field + method to `StageSpec` at `:167-204`)
- Test: `tests/test_see_also_schema.py` (create)
- Modify: `docs/Codebase.md`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SeeAlso(anchor: str, text: str)` — frozen dataclass; `anchor` is `""` (stage-level) or `"param:<name>"`. Raises `ValueError` on any other prefix and on empty `text`.
  - `StageSpec.see_also: tuple[SeeAlso, ...] = ()`
  - `StageSpec.see_also_problems() -> list[str]` — cross-reference check; empty list means valid.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_see_also_schema.py`:

```python
"""The SeeAlso pointer carrier on a stage schema (Qt-free)."""

import pytest

from dfxm.config.models import Param, ParamType, SeeAlso, StageSpec


def _spec(*see_also):
    return StageSpec(
        name="demo",
        label="Demo",
        description="A demo stage.",
        params=(
            Param("colormap", ParamType.STR, "Colormap"),
            Param("vmin", ParamType.FLOAT, "vmin"),
        ),
        see_also=see_also,
    )


def test_a_spec_has_no_pointers_by_default():
    spec = StageSpec(name="d", label="D", description="d", params=())
    assert spec.see_also == ()
    assert spec.see_also_problems() == []


def test_a_stage_level_pointer_is_valid():
    spec = _spec(SeeAlso("", "Colormaps live in Publication style…"))
    assert spec.see_also_problems() == []


def test_a_param_pointer_naming_a_real_param_is_valid():
    spec = _spec(SeeAlso("param:colormap", "Publication style wins here."))
    assert spec.see_also_problems() == []


def test_a_param_pointer_naming_a_missing_param_is_reported():
    spec = _spec(SeeAlso("param:nope", "text"))
    problems = spec.see_also_problems()
    assert len(problems) == 1
    assert "nope" in problems[0]


def test_every_bad_pointer_is_reported_not_just_the_first():
    spec = _spec(SeeAlso("param:nope", "a"), SeeAlso("param:alsonope", "b"))
    assert len(spec.see_also_problems()) == 2


def test_an_unknown_anchor_prefix_is_rejected_at_construction():
    with pytest.raises(ValueError, match="anchor"):
        SeeAlso("group:Appearance", "text")


def test_an_empty_text_is_rejected_at_construction():
    with pytest.raises(ValueError, match="text"):
        SeeAlso("", "   ")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_see_also_schema.py -q`
Expected: FAIL — `ImportError: cannot import name 'SeeAlso' from 'dfxm.config.models'`.

- [ ] **Step 3: Implement**

In `dfxm/config/models.py`, add above `class StageSpec`:

```python
@dataclass(frozen=True)
class SeeAlso:
    """A pointer from where a user looks to where the feature actually lives.

    ``anchor`` is ``""`` (rendered once at the top of the stage form and
    appended to the help panel's idle text) or ``"param:<name>"`` (rendered
    under that parameter's editor and appended to its help text).

    There is deliberately no group-level anchor: every ``Appearance`` /
    ``Quantities`` param is ``advanced=True``, and the form builds group
    headers only inside the collapsed "Advanced" expander — a group-anchored
    pointer would be invisible to exactly the newcomer it targets.
    """

    anchor: str
    text: str

    def __post_init__(self) -> None:
        if self.anchor != "" and not self.anchor.startswith("param:"):
            raise ValueError(
                f"see-also anchor {self.anchor!r} must be '' or 'param:<name>'"
            )
        if self.anchor.startswith("param:") and not self.anchor[len("param:") :]:
            raise ValueError("see-also anchor 'param:' names no parameter")
        if not self.text.strip():
            raise ValueError("see-also text must not be empty")

    @property
    def param_name(self) -> str:
        """The parameter this points at, or ``""`` for a stage-level pointer."""
        return self.anchor[len("param:") :] if self.anchor.startswith("param:") else ""
```

Add the field to `StageSpec` (after `estimate`):

```python
    see_also: tuple[SeeAlso, ...] = ()
```

And the method (next to `get`):

```python
    def see_also_problems(self) -> list[str]:
        """Anchors that name no existing parameter (empty list = all valid).

        Prefix validity is enforced by :class:`SeeAlso` itself; this is the
        cross-reference the dataclass cannot do on its own.
        """
        names = {p.name for p in self.params}
        return [
            f"stage {self.name!r}: see-also anchor names unknown param {s.param_name!r}"
            for s in self.see_also
            if s.param_name and s.param_name not in names
        ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_see_also_schema.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Run the mutations**

Each mutation must fail **at the named assertion line**. Use `python3 -B`.

| Mutation | Must break |
|---|---|
| `see_also_problems` returns `[]` unconditionally | `test_a_param_pointer_naming_a_missing_param_is_reported` |
| `see_also_problems` returns after the first problem (`[...][:1]`) | `test_every_bad_pointer_is_reported_not_just_the_first` |
| Delete the prefix check in `__post_init__` | `test_an_unknown_anchor_prefix_is_rejected_at_construction` |
| Delete the empty-text check | `test_an_empty_text_is_rejected_at_construction` |
| `see_also` default changed to a non-empty tuple | `test_a_spec_has_no_pointers_by_default` |

Run each as: `python3 -B -m pytest tests/test_see_also_schema.py -q`, confirm the expected test fails, then revert the mutation.

- [ ] **Step 6: Update `docs/Codebase.md`**

In the `dfxm/config/models.py` section, document `SeeAlso` (both anchor kinds, why there is no group anchor) and `StageSpec.see_also` / `see_also_problems()`.

- [ ] **Step 7: Verify and commit**

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
git add dfxm/config/models.py tests/test_see_also_schema.py docs/Codebase.md
git commit -m "feat: SeeAlso pointer carrier on StageSpec"
```

---

### Task 2: Render pointers in the form and help panel

**Files:**
- Modify: `gui/widgets/param_form.py` (constructor `:62-122`, new `_add_see_also_rows`)
- Modify: `gui/widgets/help_panel.py` (`param_help_html` `:20-42`, `HelpPanel` `:45-84`)
- Modify: `gui/theme.py` (add the `hint` role next to `:172-177`)
- Modify: `gui/stage_view.py:124` and `:212-215` (pass `spec.see_also` through)
- Test: `tests/test_gui_see_also_render.py` (create)
- Modify: `docs/Codebase.md`

**Interfaces:**
- Consumes: `SeeAlso`, `StageSpec.see_also` (Task 1).
- Produces:
  - `ParamForm(params, values=None, parent=None, see_also=())` — new keyword-only-safe trailing argument; existing positional calls are unaffected.
  - `ParamForm._see_also_labels: dict[str, QLabel]` — keyed by `""` for the stage-level row and by param name for param rows (test access, mirroring `_notes`).
  - `param_help_html(p, error_color=None, see_also="")`
  - `HelpPanel.set_see_also(mapping: dict[str, str])` — param name → pointer text.
  - `HelpPanel.set_idle(title, description, see_also="")`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gui_see_also_render.py`:

```python
"""See-also pointers rendered inline and in the help panel (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QComboBox  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.config.models import Param, ParamType, SeeAlso  # noqa: E402
from gui.widgets.help_panel import HelpPanel, param_help_html  # noqa: E402
from gui.widgets.param_form import ParamForm  # noqa: E402

_PARAMS = (
    Param("colormap", ParamType.ENUM, "Colormap", default="fast", choices=("fast", "gray")),
    Param("plain", ParamType.STR, "Plain"),
)

_STAGE_PTR = SeeAlso("", "Colormaps are set in Publication style… (left panel).")
_PARAM_PTR = SeeAlso("param:colormap", "Publication style wins for standard quantities.")


def test_a_form_without_pointers_renders_no_pointer_rows():
    form = ParamForm(_PARAMS)
    assert form._see_also_labels == {}


def test_a_stage_pointer_renders_one_always_visible_row():
    form = ParamForm(_PARAMS, see_also=(_STAGE_PTR,))
    label = form._see_also_labels[""]
    assert label.text() == _STAGE_PTR.text
    # The point of a pointer is that it needs no expanding or focusing.
    assert label.isVisibleTo(form) is True


def test_a_param_pointer_renders_under_that_param_only():
    form = ParamForm(_PARAMS, see_also=(_PARAM_PTR,))
    assert set(form._see_also_labels) == {"colormap"}
    assert form._see_also_labels["colormap"].text() == _PARAM_PTR.text
    assert form._see_also_labels["colormap"].isVisibleTo(form) is True


def test_pointer_rows_are_styled_as_hints_not_warnings():
    # Advisory notes use role="warning" and are hidden by default; pointers are
    # a different thing and must not borrow that styling.
    form = ParamForm(_PARAMS, see_also=(_STAGE_PTR, _PARAM_PTR))
    for label in form._see_also_labels.values():
        assert label.property("role") == "hint"


def test_a_pointer_does_not_wrap_the_editor_widget():
    # gui_smoke and the wheel-guard tests reach into _editors[name] directly.
    form = ParamForm(_PARAMS, see_also=(_PARAM_PTR,))
    assert isinstance(form._editors["colormap"], QComboBox)


def test_a_pointer_for_an_unknown_param_renders_nothing_rather_than_crashing():
    form = ParamForm(_PARAMS, see_also=(SeeAlso("param:ghost", "text"),))
    assert form._see_also_labels == {}


def test_param_help_html_appends_the_pointer():
    html = param_help_html(_PARAMS[0], see_also="Set in Publication style…")
    assert "See also:" in html
    assert "Set in Publication style…" in html


def test_param_help_html_without_a_pointer_has_no_see_also_line():
    assert "See also:" not in param_help_html(_PARAMS[0])


def test_the_help_panel_idle_text_carries_a_stage_pointer():
    panel = HelpPanel()
    panel.set_idle("Strain", "Compute strain maps.", see_also=_STAGE_PTR.text)
    assert "Compute strain maps." in panel._label.text()  # precondition
    assert _STAGE_PTR.text in panel._label.text()


def test_the_help_panel_shows_a_param_pointer_when_that_param_is_focused():
    panel = HelpPanel()
    panel.set_see_also({"colormap": _PARAM_PTR.text})
    panel.show_param(_PARAMS[0])
    assert _PARAM_PTR.text in panel._label.text()
    panel.show_param(_PARAMS[1])
    assert _PARAM_PTR.text not in panel._label.text()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_see_also_render.py -q`
Expected: FAIL — `ParamForm.__init__() got an unexpected keyword argument 'see_also'`.

- [ ] **Step 3: Implement the form rendering**

In `gui/widgets/param_form.py`, extend the constructor signature:

```python
    def __init__(
        self,
        params: Sequence[Param],
        values: dict[str, Any] | None = None,
        parent: QWidget | None = None,
        see_also: Sequence[SeeAlso] = (),
    ) -> None:
```

Add the import `from dfxm.config.models import Param, ParamType, SeeAlso`, and after `self._notes: dict[str, QLabel] = {}`:

```python
        self._see_also_labels: dict[str, QLabel] = {}
        self._see_also_by_param = {s.param_name: s for s in see_also if s.param_name}
        self._stage_see_also = tuple(s for s in see_also if not s.param_name)
```

Immediately after `outer.setContentsMargins(0, 0, 0, 0)`, render the stage-level pointers above everything:

```python
        for entry in self._stage_see_also:
            outer.addWidget(self._see_also_label("", entry.text))
```

Add the row builder and the per-param hook next to `_add_note_row`:

```python
    def _see_also_label(self, key: str, text: str) -> QLabel:
        """A quiet, always-visible pointer row (role="hint", never hidden).

        Distinct from `_add_note_row`'s role="warning" advisory rows, which
        start hidden and carry cost warnings: a pointer is static text whose
        whole purpose is being visible without being sought.
        """
        label = QLabel(text)
        label.setWordWrap(True)
        label.setProperty("role", "hint")
        self._see_also_labels[key] = label
        return label

    def _add_see_also_row(self, form: QFormLayout, p: Param) -> None:
        """A pointer row under *p*'s editor, when the spec declares one.

        The editor itself is NOT wrapped: `self._editors[name]` must stay the
        real widget, which `gui_smoke` and the wheel-guard tests reach into
        directly — the same constraint `_add_note_row` documents.
        """
        entry = self._see_also_by_param.get(p.name)
        if entry is None:
            return
        form.addRow(self._see_also_label(p.name, entry.text))
```

Call it after each `self._add_note_row(...)` — both the essentials loop and the advanced loop:

```python
            ess_form.addRow(self._label_for(p), self._make_editor(p, initial))
            self._add_note_row(ess_form, p)
            self._add_see_also_row(ess_form, p)
```

```python
                form.addRow(self._label_for(p), self._make_editor(p, initial))
                self._add_note_row(form, p)
                self._add_see_also_row(form, p)
```

An anchor naming no param simply never matches and renders nothing — Task 1's `see_also_problems()` is what turns that into a test failure, so it must not also crash the form.

Pass the pointer into the field tooltip in `_make_editor`, replacing `tip = param_help_html(p)`:

```python
        entry = self._see_also_by_param.get(p.name)
        tip = param_help_html(p, see_also=entry.text if entry else "")
```

- [ ] **Step 4: Implement the help panel**

In `gui/widgets/help_panel.py`, extend `param_help_html`:

```python
def param_help_html(p: Param, error_color: str | None = None, see_also: str = "") -> str:
```

and before `return "<br>".join(parts)`:

```python
    if see_also:
        parts.append(f"<i>See also:</i> {html.escape(see_also)}")
```

In `HelpPanel.__init__`, add `self._see_also: dict[str, str] = {}`. Then:

```python
    def set_idle(self, title: str, description: str, see_also: str = "") -> None:
        """Set (and show) the text used when no field is focused."""
        self._idle_html = f"<b>{html.escape(title)}</b> — {html.escape(description)}"
        if see_also:
            self._idle_html += f"<br><i>See also:</i> {html.escape(see_also)}"
        self._current = None
        self._render()

    def set_see_also(self, mapping: dict[str, str]) -> None:
        """Pointer text per parameter name, appended when that param is shown."""
        self._see_also = dict(mapping)
        self._render()
```

and in `_render`:

```python
        self._label.setText(
            param_help_html(
                self._current, self._error_color, self._see_also.get(self._current.name, "")
            )
        )
```

- [ ] **Step 5: Add the `hint` role to the theme**

In `gui/theme.py`, beside the other `QLabel[role=...]` rules:

```python
    QLabel[role="hint"] {{ color: {p.ink_muted}; font-style: italic; }}
```

- [ ] **Step 6: Wire it into `StageView`**

`gui/stage_view.py:124`:

```python
        self._form = ParamForm(spec.params, self._initial_values(), see_also=spec.see_also)
```

`gui/stage_view.py:212-213`:

```python
        self._help = HelpPanel()
        _stage_ptr = " ".join(s.text for s in spec.see_also if not s.param_name)
        self._help.set_idle(spec.label, spec.description, see_also=_stage_ptr)
        self._help.set_see_also(
            {s.param_name: s.text for s in spec.see_also if s.param_name}
        )
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_gui_see_also_render.py tests/test_gui_param_notes.py tests/test_gui_wheel_guard.py -q`
Expected: PASS — including the two existing files, which prove the new argument broke no existing construction site.

- [ ] **Step 8: Run the mutations**

| Mutation | Must break |
|---|---|
| `_see_also_label` sets `role="warning"` | `test_pointer_rows_are_styled_as_hints_not_warnings` |
| `_see_also_label` calls `setVisible(False)` | `test_a_stage_pointer_renders_one_always_visible_row` |
| `_add_see_also_row` renders for every param (drop the `is None` guard) | `test_a_param_pointer_renders_under_that_param_only` |
| Stage-level entries also added per-param (drop the `param_name` filter) | `test_a_form_without_pointers_renders_no_pointer_rows` / `..._under_that_param_only` |
| `param_help_html` appends the see-also line unconditionally | `test_param_help_html_without_a_pointer_has_no_see_also_line` |
| `HelpPanel._render` ignores `self._see_also` | `test_the_help_panel_shows_a_param_pointer_when_that_param_is_focused` |
| `set_idle` drops the `see_also` argument | `test_the_help_panel_idle_text_carries_a_stage_pointer` |

Run each with `python3 -B`, confirm the named test fails, revert.

- [ ] **Step 9: Update `docs/Codebase.md`**

Document `ParamForm(..., see_also=)` and `_see_also_label`/`_add_see_also_row`, `param_help_html`'s third argument, `HelpPanel.set_see_also` / the extended `set_idle`, and the new `hint` QSS role.

- [ ] **Step 10: Verify and commit**

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
git add gui/widgets/param_form.py gui/widgets/help_panel.py gui/theme.py gui/stage_view.py \
        tests/test_gui_see_also_render.py docs/Codebase.md
git commit -m "feat: render see-also pointers in the form and help panel"
```

---

### Task 3: The colormap pointers

Data only — the mechanism already works.

**Files:**
- Modify: `dfxm/stages/strain.py`, `mosaicity.py`, `rocking.py`, `visualize.py`, `slices.py`, `profiles.py`, `matched.py` (each stage's `SPEC` / `StageSpec(...)` call)
- Test: `tests/test_see_also_schema.py` (append)
- Modify: `docs/Usage.md`

**Interfaces:**
- Consumes: `SeeAlso`, `StageSpec.see_also`, `see_also_problems()` (Task 1); the rendering (Task 2).
- Produces: pointer text visible on seven stages. No behaviour change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_see_also_schema.py`:

```python
from gui.bindings import STAGE_ORDER, STAGE_SPECS  # noqa: E402

_FIGURE_STAGES = ("strain", "mosaicity", "rocking", "visualize", "slices", "profiles", "matched")


def test_every_real_stage_spec_has_valid_see_also_anchors():
    # Precondition: this walk is worthless if no stage declares a pointer.
    assert sum(len(STAGE_SPECS[n].see_also) for n in STAGE_ORDER) > 0
    for name in STAGE_ORDER:
        assert STAGE_SPECS[name].see_also_problems() == []


def test_every_figure_producing_stage_points_at_the_style_dialog():
    for name in _FIGURE_STAGES:
        texts = [s.text for s in STAGE_SPECS[name].see_also if not s.param_name]
        assert texts, f"{name} has no stage-level pointer"
        assert any("Publication style" in t for t in texts), name


def test_stages_that_produce_no_figures_have_no_pointer():
    # concat writes .h5 only; paraview writes VTI whose colormap is chosen in
    # ParaView itself. A pointer there would be a lie.
    for name in ("concat", "paraview"):
        assert STAGE_SPECS[name].see_also == ()


def test_matched_additionally_annotates_its_colormap_dropdown():
    entries = {s.param_name: s.text for s in STAGE_SPECS["matched"].see_also if s.param_name}
    assert "colormap" in entries
    assert "Publication style" in entries["colormap"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_see_also_schema.py -q`
Expected: FAIL — the precondition assertion (`sum(...) > 0`) and `test_every_figure_producing_stage_points_at_the_style_dialog`.

- [ ] **Step 3: Add the pointers**

Read each stage module's `StageSpec(...)` call before editing it. Add a `see_also=` argument after `params=(...)`.

For **strain, mosaicity, rocking, visualize, slices, profiles** — one stage-level entry each, wording adjusted to the stage's own fields:

```python
    see_also=(
        SeeAlso(
            "",
            "Colormaps are set per quantity group in “Publication style…” "
            "(left panel), not here.",
        ),
    ),
```

For **strain**, **rocking** and **matched**, which do have their own range fields, extend the sentence so the pointer explains the split rather than contradicting the form:

```python
            "Colormaps are set per quantity group in “Publication style…” "
            "(left panel); the range fields below are this stage's own.",
```

For **matched**, add the second, param-anchored entry as well:

```python
    see_also=(
        SeeAlso(
            "",
            "Colormaps are set per quantity group in “Publication style…” "
            "(left panel); the range fields below are this stage's own.",
        ),
        SeeAlso(
            "param:colormap",
            "“Publication style…” drives the raw-intensity colormap; this is "
            "the fallback for anything without a quantity group.",
        ),
    ),
```

Each module needs `SeeAlso` added to its existing `from dfxm.config.models import ...` line.

Leave `concat` and `paraview` untouched.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_see_also_schema.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Run the mutations**

| Mutation | Must break |
|---|---|
| Delete `see_also=` from `strain.py`'s SPEC | `test_every_figure_producing_stage_points_at_the_style_dialog` |
| Change one pointer's anchor to `"param:ghost"` | `test_every_real_stage_spec_has_valid_see_also_anchors` |
| Add a `see_also` entry to `concat.py` | `test_stages_that_produce_no_figures_have_no_pointer` |
| Delete matched's `param:colormap` entry | `test_matched_additionally_annotates_its_colormap_dropdown` |
| Delete **every** stage's `see_also` | the precondition line in `test_every_real_stage_spec_has_valid_see_also_anchors` — confirm it is that line, not the loop |

- [ ] **Step 6: Look at it**

```bash
DISPLAY= QT_QPA_PLATFORM=offscreen python3 -c "
from PySide6.QtWidgets import QApplication
app = QApplication([])
from gui.bindings import STAGE_SPECS
from gui.widgets.param_form import ParamForm
for n in ('strain','matched'):
    s = STAGE_SPECS[n]
    f = ParamForm(s.params, s.defaults(), see_also=s.see_also)
    print(n, {k: v.text() for k, v in f._see_also_labels.items()})
"
```

Confirm each stage prints the pointer(s) you expect, keyed as expected.

- [ ] **Step 7: Update `docs/Usage.md`**

In the section covering publication style / per-stage appearance, state that every figure-producing stage's form now carries a pointer to *Publication style…*, and that matched's own `Colormap` field is the fallback for quantities with no group (forward-reference Task 4's behaviour change; Task 4 completes the sentence).

- [ ] **Step 8: Verify and commit**

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
git add dfxm/stages/*.py tests/test_see_also_schema.py docs/Usage.md
git commit -m "feat: point every figure stage at the publication style dialog"
```

---

### Task 4: Group-wire `matched` to the `raw` quantity

The one real behaviour change in this plan.

**Files:**
- Modify: `dfxm/stages/matched.py:584-594` (`run`), `:665-675` (export rebuild), `:615`/`:650` (the recorded fallback)
- Test: `tests/test_stage_matched.py` (append)
- Modify: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `dfxm.common.plotting.resolve_cmap`, `render.layer_figure(..., group=)`.
- Produces: matched figures resolved through the session style. `MatchedLayer.colormap` keeps its current meaning (the stage's fallback) — the dataclass is unchanged, so no stored result breaks.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage_matched.py`. It already has `_write_strain` /
`_write_rocking` helpers and `NF, H, W` at module level — reuse them:

```python
from dataclasses import asdict

from dfxm.common.plotting import PlotStyle


def _one_layer_setup(tmp_path):
    """The smallest input run() will process: one matched strain/rocking pair."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_strain(str(raw), "strain__1", 0.0, 0.0)
    frames = np.random.default_rng(0).standard_normal((NF, H, W)) + 10.0
    _write_rocking(str(raw), "rock__1", 0.0, 0.0, frames)
    return {
        "raw_root": str(raw),
        "strain_pattern": "strain__*",
        "rocking_pattern": "rock__*",
        "frame_index": 0,
        "match_threshold_mm": 0.001,
        "output_dir": str(tmp_path / "out"),
        "colormap": "gray",
        "plot_style": asdict(PlotStyle(cmap_raw="turbo")),
    }


def _capture_layer_figure(monkeypatch):
    """Record the cmap/group handed to render.layer_figure, still rendering."""
    seen = {}
    real = M.Rnd.layer_figure

    def spy(layer, vmin, vmax, cmap, ex, ey, title, cbar, *, style=None, group=None):
        seen["cmap"] = cmap
        seen["group"] = group
        return real(layer, vmin, vmax, cmap, ex, ey, title, cbar, style=style, group=group)

    monkeypatch.setattr(M.Rnd, "layer_figure", spy)
    return seen


def test_run_renders_the_raw_quantity_group(tmp_path, monkeypatch):
    """matched draws rocking-curve frames — the "raw" group, like rocking.py."""
    params = _one_layer_setup(tmp_path)
    assert params["colormap"] == "gray"  # precondition: the fallback differs
    seen = _capture_layer_figure(monkeypatch)
    res = M.run(params)
    assert res.n_saved == 1  # precondition: a layer was actually drawn
    assert seen["group"] == "raw"
    assert seen["cmap"] == "turbo"  # the style's raw cmap, not the param


def test_the_export_rebuild_resolves_the_same_colormap_as_the_run(tmp_path, monkeypatch):
    """An exported figure must match the PNG the run saved."""
    params = _one_layer_setup(tmp_path)
    res = M.run(params)
    assert res.recorded and res.recorded[0].colormap == "gray"  # precondition
    specs = M.figures(res, params)
    assert specs  # precondition: there is something to rebuild
    seen = _capture_layer_figure(monkeypatch)
    specs[0].build(PlotStyle(cmap_raw="turbo"))
    assert seen["cmap"] == "turbo"
    assert seen["group"] == "raw"


def test_the_stage_colormap_is_still_the_fallback_without_a_group():
    from dfxm.common.plotting import resolve_cmap

    assert resolve_cmap(PlotStyle(cmap_raw="turbo"), None, fallback="magma") == "magma"
```

`M.run` reads the style via `style_from_params(p)` (`matched.py:465`), which is
why the style travels in as `params["plot_style"]`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_matched.py -q`
Expected: FAIL — `test_run_renders_the_raw_quantity_group` fails at
`assert seen["group"] == "raw"` (it is `None`), and the rebuild test fails at
`assert seen["cmap"] == "turbo"` (it is `"gray"`).

- [ ] **Step 3: Implement**

In `run()` (`:584-594`), replace the positional `p["colormap"]`:

```python
        fig, _, _ = Rnd.layer_figure(
            shifted,
            vmin,
            vmax,
            resolve_cmap(style, "raw", fallback=p["colormap"]),
            ext_x,
            ext_y,
            title,
            "Intensity − background (a.u.)",
            style=style,
            group="raw",
        )
```

In the export rebuild's `build(style)` (`:665-675`), identically:

```python
            fig, _, _ = Rnd.layer_figure(
                shifted,
                vmin,
                vmax,
                resolve_cmap(style, "raw", fallback=colormap),
                ext_x,
                ext_y,
                title,
                "Intensity − background (a.u.)",
                style=style,
                group="raw",
            )
```

Add `resolve_cmap` to the module's `dfxm.common.plotting` import. Leave `MatchedLayer.colormap` and both `colormap=`/`rec.colormap` assignments alone — the recorded value is now the fallback, and the NOTE at `:599-601` about `run()` and `figures().build()` staying in sync still holds and now covers the group too. Extend that NOTE to say so.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_matched.py -q`
Expected: PASS.

- [ ] **Step 5: Run the mutations**

| Mutation | Must break |
|---|---|
| Drop `group="raw"` from `run()` | `test_run_renders_the_raw_quantity_group` |
| Drop `group="raw"` from `build()` only | `test_the_export_rebuild_resolves_the_same_colormap_as_the_run` |
| `resolve_cmap(style, None, fallback=...)` in `run()` | `test_run_renders_the_raw_quantity_group`, at the `seen["cmap"]` line |
| `MatchedLayer.colormap` recorded as `"turbo"` instead of the param | the `res.recorded[0].colormap == "gray"` precondition in the rebuild test |

- [ ] **Step 6: Update both docs**

- `docs/Usage.md` — matched's figures now follow the *Publication style…* `raw` colormap (and its tick format); the stage's own `Colormap` field remains as the fallback. **Say plainly that matched figures will look different on the next run than they did before this change.**
- `docs/Codebase.md` — matched's `run()`/`figures()` resolve through `resolve_cmap(style, "raw", ...)`, mirroring `rocking.py`.

- [ ] **Step 7: Verify and commit**

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
git add dfxm/stages/matched.py tests/test_stage_matched.py docs/Usage.md docs/Codebase.md
git commit -m "fix: matched follows the publication style's raw colormap"
```

---

### Task 5: Style-at-Run signalling

**Files:**
- Modify: `gui/stage_view.py:429-437` (capture), `:884-893` (stamp)
- Modify: `gui/main_window.py:231-255` (dialog note)
- Test: `tests/test_gui_style_stamp.py` (create)
- Modify: `tests/gui_smoke.py` (new step `[44]`)
- Modify: `docs/Usage.md`

**Interfaces:**
- Consumes: `MainWindow.global_plot_style()`, `dfxm.common.plotting.PlotStyle`.
- Produces:
  - `gui.stage_view.style_stamp(style: PlotStyle | None) -> str` — module-level, pure, returns `""` for `None`.
  - `StageView._last_style: PlotStyle | None` — the style the last run was launched with.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gui_style_stamp.py`:

```python
"""The publication style a finished run actually used (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.common.plotting import PlotStyle  # noqa: E402
from gui.stage_view import style_stamp  # noqa: E402


def test_no_style_stamps_nothing():
    assert style_stamp(None) == ""


def test_the_stamp_names_all_four_group_colormaps_and_the_font_scale():
    style = PlotStyle(
        cmap_mosa_com="fast",
        cmap_mosa_fwhm="magma",
        cmap_strain="RdBu_r",
        cmap_raw="gray",
        font_scale=1.25,
    )
    stamp = style_stamp(style)
    for expected in ("fast", "magma", "RdBu_r", "gray", "1.25"):
        assert expected in stamp, expected


def test_the_stamp_says_it_is_the_style_the_run_used():
    stamp = style_stamp(PlotStyle())
    assert "rendered with" in stamp.lower()


def test_the_stamp_reflects_the_captured_style_not_the_current_one():
    at_launch = PlotStyle(cmap_strain="turbo")
    edited_since = PlotStyle(cmap_strain="seismic")
    assert "turbo" in style_stamp(at_launch)
    assert "seismic" not in style_stamp(at_launch)  # precondition: they differ
    assert "seismic" in style_stamp(edited_since)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_style_stamp.py -q`
Expected: FAIL — `ImportError: cannot import name 'style_stamp'`.

- [ ] **Step 3: Implement the stamp**

Add to `gui/stage_view.py` at module level (next to `_summarize`):

```python
def style_stamp(style) -> str:
    """One line naming the publication style a run rendered with.

    Runs snapshot the style at launch (`_on_run`), so a style edited afterwards
    does not retro-apply to a finished run — a fact users reasonably expect to
    work the other way. Recording it on the result is how they can tell.
    """
    if style is None:
        return ""
    return (
        "Rendered with publication style: "
        f"mosa_com={style.cmap_mosa_com}, mosa_fwhm={style.cmap_mosa_fwhm}, "
        f"strain={style.cmap_strain}, raw={style.cmap_raw}, "
        f"font ×{style.font_scale:g}"
    )
```

Capture the style at launch — in `_on_run`, inside the existing `if hasattr(window, "global_plot_style"):` block at `:432-437`:

```python
            self._last_style = window.global_plot_style()
            run_params["plot_style"] = asdict(self._last_style)
```

Initialise `self._last_style = None` beside `self._last_result`.

Append it in `_finish_ok`, replacing `:890-893`:

```python
        summary = _summarize(self._stage_name, result)
        first_line = summary.splitlines()[0] if summary else "done"
        self._show_banner(f"✓ {html.escape(first_line)}", error=False)
        stamp = style_stamp(self._last_style)
        self._results.setPlainText(f"{summary}\n\n{stamp}" if stamp else summary)
```

The banner keeps using `first_line` — the stamp belongs in the Results tab, not in the one-line banner.

- [ ] **Step 4: Add the dialog note**

In `gui/main_window.py._on_pub_style`, between the scroll area and the button box:

```python
        note = QLabel(
            "These settings apply to runs started from now on — a finished run "
            "keeps the style it was launched with. Use “Replot…” on a stage to "
            "re-render finished results with the current style."
        )
        note.setWordWrap(True)
        note.setProperty("role", "hint")

        layout = QVBoxLayout(dlg)
        layout.addWidget(scroll, 1)
        layout.addWidget(note)
        layout.addWidget(btn_box)
```

`QLabel` is already imported in this module.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_gui_style_stamp.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Add smoke step `[44]`**

In `tests/gui_smoke.py`, after step `[43]`, following the file's existing style: build a `StageView`, set `_last_style` to a `PlotStyle(cmap_strain="turbo")`, call `_finish_ok` with the stage's minimal result object (copy the pattern the neighbouring steps use), and assert `"turbo" in view._results.toPlainText()`. Print `"[44] style stamp: finished run records the style it used"`.

`_results` **is** a `QPlainTextEdit` and does have `toPlainText()`. `_log` is a `LogConsole` and does **not** — do not reach for it here.

- [ ] **Step 7: Run the mutations**

| Mutation | Must break |
|---|---|
| `style_stamp` returns `""` always | `test_the_stamp_names_all_four_group_colormaps_and_the_font_scale` |
| Drop `cmap_raw` from the stamp | same test — confirm it fails on the `"gray"` iteration |
| `_finish_ok` stamps `window.global_plot_style()` instead of `self._last_style` | smoke `[44]` |
| Drop the "Rendered with" prefix | `test_the_stamp_says_it_is_the_style_the_run_used` |

- [ ] **Step 8: Update `docs/Usage.md`**

Document that the Results tab records the style a run used, and that the Publication style dialog states the timing rule and points at `Replot…`.

- [ ] **Step 9: Verify and commit**

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
DISPLAY= python3 -u tests/gui_smoke.py
ruff check . && ruff format .
git add gui/stage_view.py gui/main_window.py tests/test_gui_style_stamp.py \
        tests/gui_smoke.py docs/Usage.md
git commit -m "feat: record the publication style each run rendered with"
```

---

### Task 6: A summary editor for `jobs_json`

**Files:**
- Modify: `dfxm/config/models.py` (add `Param.editor`)
- Create: `gui/widgets/jobs_summary.py`
- Modify: `gui/widgets/param_form.py` (`_build_editor` dispatch `:269-282`)
- Modify: `dfxm/stages/profiles.py:148-159` (declare the hint)
- Test: `tests/test_gui_jobs_summary.py` (create), `tests/test_param_metadata.py` (append)
- Modify: `tests/gui_smoke.py` (new step `[45]`)
- Modify: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `ParamType.TEXT`.
- Produces:
  - `Param.editor: str = ""` — a render hint; `"summary_json"` is the only value the form understands today. An unknown value falls back to the type's normal editor.
  - `gui.widgets.jobs_summary.summarize_jobs(raw: str) -> str` — pure; never raises.
  - `gui.widgets.jobs_summary.JobsSummaryEditor(value: str, label: str)` with `text() -> str`, `setText(str)`, and a `textChanged` signal — the same trio `ParamForm._register` needs.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gui_jobs_summary.py`:

```python
"""Summary editor for a JSON-list TEXT param (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.config.models import Param, ParamType  # noqa: E402
from gui.widgets.jobs_summary import JobsSummaryEditor, summarize_jobs  # noqa: E402
from gui.widgets.param_form import ParamForm  # noqa: E402

_TWO_JOBS = """
[{"name": "oblique_full", "offset_um": 0.0},
 {"name": "ridge", "offset_um": 12.5}]
"""


def test_a_summary_names_each_job_and_its_offset():
    text = summarize_jobs(_TWO_JOBS)
    assert "2 jobs" in text
    assert "oblique_full" in text
    assert "ridge" in text
    assert "12.5" in text


def test_one_job_is_singular():
    assert summarize_jobs('[{"name": "only", "offset_um": 0.0}]').startswith("1 job")


def test_an_empty_list_says_so():
    assert "no jobs" in summarize_jobs("[]").lower()


def test_a_list_without_names_falls_back_to_a_count():
    # The widget must not assume the DFXM job schema — it is a generic
    # JSON-list editor that happens to be used by profiles.
    assert summarize_jobs("[1, 2, 3]") == "3 entries"


def test_malformed_json_reports_rather_than_raising():
    assert "unreadable" in summarize_jobs("{not json").lower()


def test_the_editor_round_trips_the_raw_string_unchanged():
    raw = _TWO_JOBS.strip()
    editor = JobsSummaryEditor(raw, "Jobs (JSON)")
    assert editor.text() == raw


def test_set_text_refreshes_the_summary():
    editor = JobsSummaryEditor("[]", "Jobs (JSON)")
    assert "no jobs" in editor._summary.text().lower()  # precondition
    editor.setText(_TWO_JOBS)
    assert "oblique_full" in editor._summary.text()


def test_set_text_emits_text_changed():
    editor = JobsSummaryEditor("[]", "Jobs (JSON)")
    seen = []
    editor.textChanged.connect(seen.append)
    editor.setText("[1]")
    assert seen


def test_a_form_uses_the_summary_editor_only_when_the_hint_asks_for_it():
    hinted = Param("jobs_json", ParamType.TEXT, "Jobs", default="[]", editor="summary_json")
    plain = Param("notes", ParamType.TEXT, "Notes", default="")
    form = ParamForm((hinted, plain))
    assert isinstance(form._editors["jobs_json"], JobsSummaryEditor)
    assert not isinstance(form._editors["notes"], JobsSummaryEditor)


def test_the_form_reads_and_writes_the_raw_string_through_the_summary_editor():
    # This is the contract that keeps the two picker call sites in stage_view
    # (_on_pick_line, _on_jobs_from_marks) working untouched.
    p = Param("jobs_json", ParamType.TEXT, "Jobs", default="[]", editor="summary_json")
    form = ParamForm((p,))
    form.set_values({"jobs_json": _TWO_JOBS})
    assert form.values()["jobs_json"] == _TWO_JOBS


def test_an_unknown_editor_hint_falls_back_to_the_normal_editor():
    p = Param("notes", ParamType.TEXT, "Notes", default="", editor="nonesuch")
    form = ParamForm((p,))
    assert not isinstance(form._editors["notes"], JobsSummaryEditor)
```

Append to `tests/test_param_metadata.py`:

```python
def test_param_editor_hint_defaults_off_and_is_settable():
    assert Param("x", ParamType.STR, "X").editor == ""
    assert Param("x", ParamType.TEXT, "X", editor="summary_json").editor == "summary_json"


def test_profiles_declares_the_summary_editor_for_its_job_list():
    from gui.bindings import STAGE_SPECS

    assert STAGE_SPECS["profiles"].get("jobs_json").editor == "summary_json"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_jobs_summary.py tests/test_param_metadata.py -q`
Expected: FAIL — no `gui.widgets.jobs_summary` module.

- [ ] **Step 3: Add the schema hint**

In `dfxm/config/models.py`, add to `Param` after `advice_key`:

```python
    editor: str = ""  # render hint: "" = by type; "summary_json" = summary + raw dialog
```

and extend the class docstring: `editor` names a non-default editor for a param whose normal widget is a poor fit — `"summary_json"` renders a TEXT param holding a JSON list as a one-line summary with the raw text behind a button.

- [ ] **Step 4: Write the widget**

Create `gui/widgets/jobs_summary.py`:

```python
"""Summary editor for a TEXT param whose value is a JSON list.

The raw JSON stays the value — `text()` returns the exact string the form
stores — but the form shows a one-line summary instead of a wall of JSON, with
the raw text one click away. Nothing here knows about profile jobs
specifically: it reads `name`/`offset_um` when a list of objects happens to
carry them and falls back to a plain count otherwise.
"""

from __future__ import annotations

import json

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def summarize_jobs(raw: str) -> str:
    """A one-line description of the JSON list in *raw*. Never raises."""
    try:
        items = json.loads(raw) if raw.strip() else []
    except (ValueError, TypeError):
        return "unreadable JSON — open the editor to fix it"
    if not isinstance(items, list):
        return "unreadable JSON — expected a list"
    if not items:
        return "no jobs"
    described = []
    for item in items:
        if not isinstance(item, dict) or "name" not in item:
            return f"{len(items)} entries"
        offset = item.get("offset_um")
        described.append(
            f"{item['name']} @ {offset:+g} µm" if isinstance(offset, (int, float))
            else str(item["name"])
        )
    noun = "job" if len(items) == 1 else "jobs"
    return f"{len(items)} {noun}: " + ", ".join(described)


class JobsSummaryEditor(QWidget):
    """A read-only summary plus an "Edit raw JSON…" dialog.

    Exposes `text()` / `setText()` / `textChanged` so `ParamForm._register`
    can treat it exactly like a line edit — which is what keeps the profiles
    line-picker call sites working unchanged.
    """

    textChanged = Signal(str)  # noqa: N815 - mirrors QLineEdit's signal name

    def __init__(self, value: str, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = value or ""
        self._label = label
        self._summary = QLabel(summarize_jobs(self._value))
        self._summary.setWordWrap(True)
        self._summary.setProperty("role", "muted")
        self._edit_btn = QPushButton("Edit raw JSON…")
        self._edit_btn.clicked.connect(self._on_edit)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._summary, 1)
        row.addWidget(self._edit_btn)

    def text(self) -> str:
        return self._value

    def setText(self, value) -> None:  # noqa: N802 - mirrors QLineEdit's API
        self._value = str(value)
        self._summary.setText(summarize_jobs(self._value))
        self.textChanged.emit(self._value)

    def _on_edit(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(self._label)
        dlg.resize(600, 460)
        text = QPlainTextEdit()
        text.setPlainText(self._value)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout = QVBoxLayout(dlg)
        layout.addWidget(text, 1)
        layout.addWidget(buttons)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.setText(text.toPlainText())
```

- [ ] **Step 5: Dispatch to it from the form**

In `gui/widgets/param_form.py._build_editor`, before the `ParamType.TEXT` branch:

```python
        if p.type is ParamType.TEXT and p.editor == "summary_json":
            return self._summary_json_editor(p, value)
```

and add:

```python
    def _summary_json_editor(self, p: Param, value: Any) -> QWidget:
        from .jobs_summary import JobsSummaryEditor

        ed = JobsSummaryEditor("" if value is None else str(value), p.label)
        self._register(p.name, ed.text, ed.setText, ed.textChanged)
        return ed
```

An unknown `editor` value never matches this branch, so it falls through to the type's normal editor.

- [ ] **Step 6: Declare the hint on profiles**

Read `dfxm/stages/profiles.py:148-159` first, then add `editor="summary_json",` to that `Param(...)` call. Leave the `help` text alone except to change the trailing sentence to `"Easiest filled by 'Pick line…'; 'Edit raw JSON…' opens the full list."`.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_gui_jobs_summary.py tests/test_param_metadata.py -q`
Expected: PASS.

- [ ] **Step 8: Prove the picker call sites still work**

Run the existing profiles GUI tests, which exercise `set_values({"jobs_json": ...})`:

Run: `python3 -m pytest -q -k "profiles or pick_line or jobs" --deselect tests/test_gui_viewer3d.py`
Expected: PASS with **no** change to `gui/stage_view.py`. If any of them needed a change, stop — the contract in Step 4 is wrong, not the test.

- [ ] **Step 9: Add smoke step `[45]`**

In `tests/gui_smoke.py` after `[44]`: open the profiles `StageView`, assert its `_form._editors["jobs_json"]` is a `JobsSummaryEditor`, call `setText` with a two-job list, assert the summary label names both jobs and that `_form.values()["jobs_json"]` is the raw string. Do not call `_on_edit` (it opens a modal). Print `"[45] jobs summary editor: summary tracks the raw JSON"`.

- [ ] **Step 10: Run the mutations**

| Mutation | Must break |
|---|---|
| `summarize_jobs` returns `f"{len(items)} entries"` always | `test_a_summary_names_each_job_and_its_offset` |
| Drop the `try/except` in `summarize_jobs` | `test_malformed_json_reports_rather_than_raising` |
| `setText` skips `textChanged.emit` | `test_set_text_emits_text_changed` |
| `setText` skips the summary refresh | `test_set_text_refreshes_the_summary` |
| `_build_editor` uses the summary editor for every TEXT param | `test_a_form_uses_the_summary_editor_only_when_the_hint_asks_for_it` |
| `text()` returns the summary instead of the raw value | `test_the_form_reads_and_writes_the_raw_string_through_the_summary_editor` |
| Drop `editor="summary_json"` from profiles | `test_profiles_declares_the_summary_editor_for_its_job_list` |

- [ ] **Step 11: Update both docs**

- `docs/Usage.md` — the profiles form shows a job summary with `Edit raw JSON…`; the pickers still fill it.
- `docs/Codebase.md` — `Param.editor`, `gui/widgets/jobs_summary.py` (`summarize_jobs`, `JobsSummaryEditor` and its `text`/`setText`/`textChanged` contract), the `_build_editor` dispatch.

- [ ] **Step 12: Verify and commit**

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
DISPLAY= python3 -u tests/gui_smoke.py
ruff check . && ruff format .
git add dfxm/config/models.py dfxm/stages/profiles.py gui/widgets/jobs_summary.py \
        gui/widgets/param_form.py tests/test_gui_jobs_summary.py \
        tests/test_param_metadata.py tests/gui_smoke.py docs/Usage.md docs/Codebase.md
git commit -m "feat: summarize the profiles job list instead of showing raw JSON"
```

---

### Task 7: Tooltips for the buttons that have none

**Files:**
- Modify: `gui/main_window.py:119-132`
- Modify: `gui/stage_view.py:139-176` (action buttons), `:236-246` and `:920-921` (export buttons)
- Modify: `gui/stage_view.py:254-262` (3-D tab tooltip)
- Test: `tests/test_gui_tooltips.py` (create)
- Modify: `docs/Usage.md`

**Interfaces:**
- Consumes: nothing.
- Produces: no new API. `StageView` gains two module constants, `EXPORT_TIP_DISABLED` and `EXPORT_TIP_ENABLED`, so the wording is pinned by test rather than duplicated as a literal.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gui_tooltips.py`:

```python
"""Every action button explains itself (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from gui.bindings import STAGE_SPECS  # noqa: E402
from gui.stage_view import EXPORT_TIP_DISABLED, EXPORT_TIP_ENABLED, StageView  # noqa: E402


def _view(stage):
    return StageView(stage, STAGE_SPECS[stage])


def test_every_stage_action_button_has_a_tooltip():
    for stage in ("profiles", "slices", "strain"):
        view = _view(stage)
        buttons = [
            view._pick_btn,
            view._jobs_marks_btn,
            view._replot_btn,
            view._pin_btn,
            view._mark_btn,
            *view._roi_buttons.values(),
        ]
        present = [b for b in buttons if b is not None]
        assert present, f"{stage} builds no action buttons"  # precondition
        for btn in present:
            assert btn.toolTip().strip(), f"{stage}: {btn.text()!r} has no tooltip"


def test_the_export_buttons_explain_why_they_are_disabled():
    view = _view("strain")
    assert view._export_btn.isEnabled() is False  # precondition
    assert view._export_btn.toolTip() == EXPORT_TIP_DISABLED
    assert "run" in EXPORT_TIP_DISABLED.lower()


def test_the_export_tooltip_changes_once_a_run_has_produced_figures():
    view = _view("strain")
    assert view._export_btn.toolTip() == EXPORT_TIP_DISABLED  # precondition
    view._enable_exports()
    assert view._export_btn.isEnabled() is True
    assert view._export_btn.toolTip() == EXPORT_TIP_ENABLED
    assert view._export_all_btn.toolTip() == EXPORT_TIP_ENABLED


def test_the_3d_tab_carries_a_tooltip_on_volume_stages():
    view = _view("visualize")
    idx = [view._tabs.tabText(i) for i in range(view._tabs.count())].index("3D")
    assert view._tabs.tabToolTip(idx).strip()


def test_the_left_panel_buttons_explain_themselves():
    from gui.main_window import MainWindow

    win = MainWindow()
    assert win._pub_style_btn.toolTip().strip()
    assert win._figure_builder_btn.toolTip().strip()
    assert "colormap" in win._pub_style_btn.toolTip().lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_tooltips.py -q`
Expected: FAIL — `ImportError: cannot import name 'EXPORT_TIP_DISABLED'`.

- [ ] **Step 3: Implement in `stage_view.py`**

Module constants next to `_VOLUME_STAGES`:

```python
EXPORT_TIP_DISABLED = "Available once a run has produced figures."
EXPORT_TIP_ENABLED = "Save figures from the last run as PNG/PDF/SVG."
```

Set a tooltip as each action button is built — the wording each button needs:

```python
            self._pick_btn.setToolTip(
                "Draw a profile line on a slice; writes it into the job list."
            )
            self._jobs_marks_btn.setToolTip(
                "Build profile jobs from the planes starred in the slices stage."
            )
            self._replot_btn.setToolTip(
                "Re-render figures from an existing .h5 without re-running the stage."
            )
            self._pin_btn.setToolTip(
                "Pin sweep planes so later runs re-render only those."
            )
            self._mark_btn.setToolTip(
                "Star interesting planes; other stages can pick them up from /marks."
            )
            _btn.setToolTip(
                "Draw the region of interest on a preview instead of typing pixel bounds."
            )
```

Export buttons at `:237-242`:

```python
        self._export_btn = QPushButton("Export…")
        self._export_btn.setEnabled(False)
        self._export_btn.setToolTip(EXPORT_TIP_DISABLED)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._export_all_btn = QPushButton("Export all…")
        self._export_all_btn.setEnabled(False)
        self._export_all_btn.setToolTip(EXPORT_TIP_DISABLED)
        self._export_all_btn.clicked.connect(self._on_export_all_clicked)
```

Replace `:920-921` with a call to a new method, so the enable and the wording can never drift apart:

```python
    def _enable_exports(self) -> None:
        """Enable the export buttons and switch them off the disabled wording."""
        for btn in (self._export_btn, self._export_all_btn):
            btn.setEnabled(True)
            btn.setToolTip(EXPORT_TIP_ENABLED)
```

The 3-D tab at `:260-262`:

```python
        if stage_name in _VOLUME_STAGES:
            self._vol3d = Volume3DPanel()
            idx = self._tabs.addTab(self._vol3d, "3D")
            self._tabs.setTabToolTip(
                idx,
                "Interactive 3-D view of this stage's volumes — run the stage, "
                "then pick a volume and click “Open 3D viewer…”.",
            )
```

- [ ] **Step 4: Implement in `main_window.py`**

```python
        self._pub_style_btn = QPushButton("Publication style…")
        self._pub_style_btn.setToolTip(
            "Fonts, scale bars and the per-quantity colormaps used by every "
            "stage's figures."
        )
```

```python
        self._figure_builder_btn = QPushButton("Figure builder…")
        self._figure_builder_btn.setToolTip(
            "Compose multi-panel figures from any stage's outputs."
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_gui_tooltips.py -q`
Expected: PASS (5 tests).

- [ ] **Step 6: Run the mutations**

| Mutation | Must break |
|---|---|
| Remove the tooltip from `_replot_btn` | `test_every_stage_action_button_has_a_tooltip` |
| `_enable_exports` enables but keeps the disabled wording | `test_the_export_tooltip_changes_once_a_run_has_produced_figures` |
| `EXPORT_TIP_DISABLED` reworded to omit "run" | `test_the_export_buttons_explain_why_they_are_disabled` |
| Remove `setTabToolTip` | `test_the_3d_tab_carries_a_tooltip_on_volume_stages` |
| `_pub_style_btn` tooltip reworded to omit colormaps | `test_the_left_panel_buttons_explain_themselves` |
| Build a stage with no action buttons in the loop | the precondition line in `test_every_stage_action_button_has_a_tooltip` |

- [ ] **Step 7: Update `docs/Usage.md`**

Note in the stage-view section that every action button and the 3-D tab carry hover help, and that the export buttons say what enables them.

- [ ] **Step 8: Verify and commit**

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
git add gui/stage_view.py gui/main_window.py tests/test_gui_tooltips.py docs/Usage.md
git commit -m "feat: hover help for the action buttons and the 3-D tab"
```

---

### Task 8: Final verification

**Files:** none (verification only, plus any fix a check forces).

- [ ] **Step 1: Full suite**

Run: `python3 -m pytest -q --deselect tests/test_gui_viewer3d.py`
Expected: PASS. Record the pass/skip/xfail counts; the baseline at `311829d` is **1502 passed / 13 skipped / 17 xfailed**, and this plan adds roughly 40 tests.

- [ ] **Step 2: Lint**

Run: `ruff check . && ruff format .`
Expected: clean, no reformatting.

- [ ] **Step 3: Smoke, three times**

Run: `DISPLAY= python3 -u tests/gui_smoke.py` — three times.
Expected: `[1]`–`[45]` pass. Step `[41]` is intermittent **on an unmodified tree**; a single `[41]` failure is not evidence against this branch. Steps `[44]` and `[45]` must pass every time.

- [ ] **Step 4: Cross-check against the spec**

Read `docs/superpowers/specs/2026-08-24-in-app-discoverability-design.md` and confirm each of §2.1–§2.6 has landed, plus §4's docs contract. Note explicitly that the spec's group-anchor design was **superseded** (commit `311829d`) by stage-level anchoring.

- [ ] **Step 5: Look at the real thing**

Launch `python3 -m gui.app` and walk: strain (stage pointer visible without expanding Advanced), matched (both pointers; Advanced expanded for the dropdown note), profiles (job summary + `Edit raw JSON…` round trip), a finished run's Results tab (style stamp), the Publication style dialog (timing note), the 3-D tab tooltip, and a disabled `Export…`.

- [ ] **Step 6: Hand back for the eyeball**

Report to the user: what landed, the suite/smoke numbers, that matched's figures will look different on the next run, and that the pass is additive to the GUI's visual weight so it wants an on-screen check before being called done.

---

## Self-Review

**Spec coverage:** §1 mechanism → Tasks 1–2. §2.1 colormap pointers → Task 3. §2.2 matched → Task 4. §2.3 style-at-Run → Task 5. §2.4 jobs_json → Task 6. §2.5 3-D tab → Task 7 (Step 3). §2.6 button tooltips → Task 7. §3 testing → every task's mutation step, plus Task 8. §4 docs → each task's docs step. The spec's "no group anchor" revision is honoured in Tasks 1–3.

**Type consistency:** `SeeAlso(anchor, text)` and `param_name` (Task 1) are used unchanged in Tasks 2–3. `ParamForm(..., see_also=)` (Task 2) is called by `StageView` in Task 2 Step 6 and exercised in Task 3. `param_help_html(p, error_color, see_also)` keeps its first two positional parameters, so `param_form.py:238` and `:263` still work. `style_stamp` (Task 5) and `summarize_jobs` / `JobsSummaryEditor.text/setText/textChanged` (Task 6) match their test usage. `EXPORT_TIP_DISABLED` / `EXPORT_TIP_ENABLED` / `_enable_exports` (Task 7) match the tooltip tests.

**Placeholder scan:** clean. Task 4's tests were sketched on the first pass and
have been replaced with runnable fixtures built on the `_write_strain` /
`_write_rocking` helpers already in `tests/test_stage_matched.py`.
