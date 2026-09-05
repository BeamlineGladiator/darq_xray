# User-facing documentation overhaul — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `README.md` and `docs/Usage.md` readable and correct for a
beamline scientist who has DFXM data but is not a Python developer, and make
all three shipped docs render properly on GitHub.

**Architecture:** Four sequential tasks. Task 1 lands a machine check
(`tests/test_docs_links.py`) and the mechanical GitHub-compatibility sweep
together, so the sweep is verified rather than eyeballed. Tasks 2 and 3 rewrite
the two user-facing docs on top of a now-correct link substrate. Task 4 adds
plain-language openers through `Usage.md`'s reference body and does the final
whole-suite verification. No code under `darq_xray/` is touched, and
`pyproject.toml` is not modified.

**Tech Stack:** Markdown (GitHub-flavoured, still Obsidian-valid), Python 3
throwaway script for the sweep, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-user-facing-docs-overhaul-design.md`

**Branch:** `docs/user-facing-overhaul` (already created; the spec is committed
there as `d2f6834`). Do not work on `master`.

## Global Constraints

- **Never break `tests/test_docs_dependencies.py`.** The
  `<!-- deps:start -->` / `<!-- deps:end -->` markers, and the exact
  backticked names between them, must survive in **both** `README.md` and
  `docs/Usage.md`. The list is, verbatim:
  `` `numpy` ``, `` `h5py` ``, `` `scipy` ``, `` `matplotlib` ``,
  `` `pyyaml` ``, `` `psutil` ``, `` `PySide6` ``, `` `pyvista` ``,
  `` `pyvistaqt` ``, `` `vtk` ``.
- **Never break `tests/test_docs_stage_params.py`.** Every `StageSpec`
  parameter must stay named — by identifier, by GUI label, or by a family form
  like `volume_opacity_<key>` — in **both** `docs/Usage.md` and
  `docs/Codebase.md`. This is why the `Usage.md` reference body (everything
  from `## Stage reference` down) is preserved in substance and only its
  wording around the edges changes. Deleting a parameter table is never
  allowed.
- **`pyproject.toml` is not modified.** The `[test]` extra stays exactly as it
  is; it merely stops being the headline install.
- **Only the five GitHub alert types** may appear in a callout: `NOTE`, `TIP`,
  `IMPORTANT`, `WARNING`, `CAUTION`, uppercase.
- **Headline install form**, used identically everywhere it appears:
  ```bash
  git clone https://github.com/BeamlineGladiator/darq_xray.git
  cd darq_xray
  python3 -m venv venv
  source venv/bin/activate          # Windows: venv\Scripts\activate
  pip install -e .
  ```
  `venv`, not `.venv`. No extras bracket, therefore no shell quoting to
  explain. `pip install -e ".[test]"` appears **only** in a development /
  contributing section.
- **The install must stay editable.** Any prose about installing must keep the
  fact that `experiments/` sits beside the package rather than inside it, so a
  non-editable install starts with an empty preset dropdown and no calibration.
- **Do not delete the psutil warning.** The `Usage.md` callout recording that a
  hand-picked install without `psutil` made the app report *8 GB free on a
  machine with 467 GB free* is kept verbatim (retyped as `> [!WARNING]`). It is
  a real incident and the reason the list is machine-checked.
- **Docs are excluded from ruff** (`extend-exclude = ["docs"]` in
  `pyproject.toml`), so Python inside Markdown fences is not reformatted. Do
  not add Python to `README.md` fences expecting formatting.
- **Read before first edit.** `README.md`, `docs/Usage.md` and
  `docs/Codebase.md` all predate this session; Read each one before its first
  Edit. Never reconstruct an `old_string` from memory — these files contain
  em-dashes, `…`, and non-breaking prose that reflows.

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `tests/test_docs_links.py` | **create** (Task 1) | Machine-checks that anchors resolve, no wiki links survive, and callout types render on GitHub |
| `docs/Usage.md` | modify (Tasks 1, 3, 4) | The user guide. Task 1 sweeps it mechanically; Task 3 replaces lines 1–103; Task 4 adds section openers |
| `docs/Codebase.md` | modify (Task 1 only) | The code reference. Mechanical sweep only — no prose changes |
| `README.md` | **rewrite** (Task 2) | The GitHub landing page |
| `docs/images/README.md` | **create** (Task 2) | Names the three screenshots to capture and where they go |
| `/tmp/claude-*/scratchpad/sweep_docs.py` | throwaway (Task 1) | The one-shot sweep. Not committed |

---

## Task 1: The link checker and the compatibility sweep

Lands the check and the fix together, because the check is what proves the
sweep converted all 92 wiki links correctly rather than most of them.

**Files:**
- Create: `tests/test_docs_links.py`
- Create (throwaway, not committed): `<scratchpad>/sweep_docs.py`
- Modify: `docs/Usage.md` (headings, 49 wiki links, 12 callout types, 9 collapsed callouts)
- Modify: `docs/Codebase.md` (43 wiki links, 1 callout type)

**Interfaces:**
- Consumes: nothing.
- Produces: `tests/test_docs_links.py` exposing `slugify(heading: str) -> str`
  and `heading_slugs(text: str) -> dict[str, str]`, which Task 2 and Task 3 use
  mentally (not by import) to predict the anchor a new heading will get.
  Also produces the renamed `docs/Usage.md` headings that Task 3's Contents
  block must link to.

### Facts this task depends on (already verified — do not re-derive)

- Every one of the 92 wiki-link targets currently resolves to a real heading,
  in both files. So a `KeyError` during the sweep means a **script bug**, not a
  pre-existing broken link — fix the script, do not paper over it.
- No two headings share text in any of the three docs, so GitHub's `-1`/`-2`
  dedupe suffix is never actually needed today. Implement it anyway.
- The 12 anchors already written as `](#…)` in `docs/Usage.md` are correct
  GitHub slugs and point at headings this task does **not** rename
  (`#replotting-strain-layers-without-re-running`,
  `#replotting-slices-without-re-running`, `#busy-indication`,
  `#pinned-planes-fast-re-runs`). They must still resolve afterwards.
- All 9 collapsed callouts in `docs/Usage.md` have the same summary line,
  `> [!note]- Advanced fields`, at lines 719, 840, 906, 1026, 1267, 1380, 1726,
  1945, 2030.
- `docs/Codebase.md` headings are **not** renamed. Its `—` and `&` headings
  produce double-hyphen slugs, which are ugly in a URL and invisible to a
  reader; its backticked identifiers in headings are the point of a code
  reference. Leave them.
- `README.md` is **not** swept by the script. Its single `[[wiki links]]`
  occurrence is inside backticks as prose about the syntax, and Task 2 rewrites
  the file wholesale anyway.

- [ ] **Step 1: Write the failing test**

Create `tests/test_docs_links.py`:

```python
"""The shipped docs must render on GitHub, not only in Obsidian.

`docs/Usage.md` and `docs/Codebase.md` were written in an Obsidian vault and
used its dialect: `[[wiki links]]` for every cross-reference, and callout types
(`[!info]`, `[!example]`) that only Obsidian knows. GitHub renders `[[...]]` as
literal text outside a GitHub *wiki*, so the entire Contents block of the user
guide arrived on the public repository as unclickable noise, and 92
cross-references went nowhere.

This is the same failure mode `test_docs_dependencies.py` and
`test_docs_stage_params.py` exist for: a documentation contract nothing checks
rots quietly. Three things are pinned here.

1. **Anchors resolve.** Every `](#anchor)` — and every `](Usage.md#anchor)`
   between the two docs — names a heading that exists, under GitHub's slug
   rules. A heading rename that orphans a link fails here rather than 404-ing
   for a reader.
2. **No wiki links survive.** Including the cross-file `[[Usage#Heading|alias]]`
   form.
3. **Callout types render.** GitHub supports exactly five alert types; anything
   else degrades to a plain blockquote with the literal `[!type]` on show.

Occurrences inside fenced code blocks are exempt throughout: a fence that
*shows* `[[wiki link]]` syntax is documentation, not a broken link.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "docs/Usage.md", "docs/Codebase.md")

#: The only alert types GitHub renders as a coloured callout. Obsidian renders
#: all five as well, so this set is the intersection, not a compromise.
GITHUB_ALERT_TYPES = frozenset({"NOTE", "TIP", "IMPORTANT", "WARNING", "CAUTION"})

_FENCE = re.compile(r"^\s*(?:```|~~~)")
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_WIKI = re.compile(r"\[\[[^\]\n]+\]\]")
_CALLOUT = re.compile(r"^>\s*\[!([A-Za-z]+)\][-+]?", re.MULTILINE)
#: `](#anchor)` and `](Other.md#anchor)`; the doc part may be empty.
_ANCHOR = re.compile(r"\]\(([A-Za-z0-9_./-]*)#([^)\s]+)\)")


def slugify(heading: str) -> str:
    """GitHub's heading -> anchor rule (`github-slugger`).

    Lowercase, drop HTML tags, drop every character that is not a word
    character, a space or a hyphen, then spaces to hyphens. `\\w` is used with
    its Unicode meaning so a non-ASCII heading keeps its letters; what this
    removes here in practice is backticks, dots, parentheses, colons, commas,
    quotes, `&`, `…` and the em-dash.
    """
    text = re.sub(r"<[^>]*>", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def heading_slugs(text: str) -> dict[str, str]:
    """Map every heading in *text* to the anchor GitHub will give it.

    A slug claimed twice gets `-1`, `-2`, … exactly as GitHub does. No doc
    currently has duplicate heading text, but a future one silently colliding
    would make link checking pass against the wrong section.
    """
    slugs: dict[str, str] = {}
    seen: dict[str, int] = {}
    for heading in _HEADING.findall(_without_fences(text)):
        base = slugify(heading)
        n = seen.get(base, 0)
        seen[base] = n + 1
        slugs[heading] = base if n == 0 else f"{base}-{n}"
    return slugs


def _without_fences(text: str) -> str:
    """*text* with every fenced code block blanked, line numbering preserved."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return "\n".join(out)


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def unresolved_anchors(rel_path: str) -> list[str]:
    """Anchors in *rel_path* that name no heading in their target document."""
    body = _without_fences(_read(rel_path))
    own_slugs = set(heading_slugs(_read(rel_path)).values())
    bad: list[str] = []
    for doc, anchor in _ANCHOR.findall(body):
        if not doc:
            valid = own_slugs
            where = rel_path
        else:
            target = doc if doc.endswith(".md") else f"{doc}.md"
            where = target if "/" in target else f"{Path(rel_path).parent}/{target}"
            if not (ROOT / where).exists():
                bad.append(f"{anchor} -> missing document {where}")
                continue
            valid = set(heading_slugs(_read(where)).values())
        if anchor not in valid:
            bad.append(f"#{anchor} (in {where})")
    return bad


@pytest.mark.parametrize("rel_path", DOCS)
def test_every_anchor_resolves_to_a_heading(rel_path):
    bad = unresolved_anchors(rel_path)
    assert not bad, (
        f"{len(bad)} link(s) in {rel_path} point at no heading:\n  "
        + "\n  ".join(bad)
        + "\n\nAnchors follow GitHub's slug rule: lowercase, punctuation "
        "dropped, spaces to hyphens. Renaming a heading renames its anchor."
    )


@pytest.mark.parametrize("rel_path", DOCS)
def test_no_obsidian_wiki_links(rel_path):
    found = _WIKI.findall(_without_fences(_read(rel_path)))
    assert not found, (
        f"{rel_path} still contains {len(found)} Obsidian wiki link(s), which "
        f"GitHub renders as literal text: {sorted(set(found))[:10]}\n"
        "Use [text](#github-slug), or [text](Other.md#github-slug) across docs."
    )


@pytest.mark.parametrize("rel_path", DOCS)
def test_callouts_use_types_github_renders(rel_path):
    used = {t.upper() for t in _CALLOUT.findall(_without_fences(_read(rel_path)))}
    unsupported = sorted(used - GITHUB_ALERT_TYPES)
    assert not unsupported, (
        f"{rel_path} uses callout type(s) GitHub does not render: {unsupported}. "
        f"GitHub supports only {sorted(GITHUB_ALERT_TYPES)} (Obsidian renders "
        "those too). Map [!info] -> [!NOTE] and [!example] -> [!TIP]."
    )


def test_the_check_can_actually_fail():
    """Guard the guard, as `test_docs_stage_params.py` does for its own.

    A bug in `slugify` that made every anchor resolve would leave the tests
    above passing vacuously. Pin the rule on cases that exercise every
    character class the docs actually contain.
    """
    assert slugify("Regions of interest: two windows, two frames") == (
        "regions-of-interest-two-windows-two-frames"
    )
    assert slugify("1. Concatenate") == "1-concatenate"
    assert slugify("Pinned planes (fast re-runs)") == "pinned-planes-fast-re-runs"
    assert slugify("`alignment.py`") == "alignmentpy"
    # GitHub drops the em-dash but keeps the spaces either side: a double hyphen.
    assert slugify("Tips & troubleshooting") == "tips--troubleshooting"

    # Two headings with *identical* text collapse to one map entry, keeping the
    # later slug. That is a limitation, not the case worth pinning; the case
    # worth pinning is distinct headings colliding on one slug.
    colliding = "# A B\n\ntext\n\n## A-B\n"
    assert sorted(heading_slugs(colliding).values()) == ["a-b", "a-b-1"]

    assert _WIKI.findall("see [[#Somewhere]]") == ["[[#Somewhere]]"]
    # A fence that *shows* the syntax is documentation, not a broken link.
    assert not _WIKI.findall(_without_fences("```\n[[#Somewhere]]\n```"))
    # A `# comment` in a bash fence is not a heading.
    assert heading_slugs("```bash\n# not a heading\n```\n\n## Real\n") == {"Real": "real"}
    assert _CALLOUT.findall("> [!info] x") == ["info"]
```

- [ ] **Step 2: Run the test to verify it fails on the current docs**

```bash
python3 -m pytest -q tests/test_docs_links.py
```

Expected: `test_no_obsidian_wiki_links` FAILS for `docs/Usage.md` (49 links) and
`docs/Codebase.md` (43 links); `test_callouts_use_types_github_renders` FAILS
for both (`['EXAMPLE', 'INFO']` and `['INFO']`).
`test_every_anchor_resolves_to_a_heading` and `test_the_check_can_actually_fail`
PASS already — the 12 existing anchors are correct, and the guard tests only
the helpers.

If `test_every_anchor_resolves_to_a_heading` fails at this point, stop: a
pre-existing broken anchor is a separate finding and must be reported, not
silently fixed by the sweep.

- [ ] **Step 3: Write the sweep script**

Write to the scratchpad (not the repo) as `sweep_docs.py`:

```python
#!/usr/bin/env python3
"""One-shot GitHub-compatibility sweep for docs/Usage.md and docs/Codebase.md.

Throwaway. Run once, review the diff, commit the docs, discard the script.
Doing this mechanically rather than by hand is what makes 92 conversions
uniform and the result reviewable as a diff.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/home/albert/Desktop/darq_xray")
USAGE = ROOT / "docs/Usage.md"
CODEBASE = ROOT / "docs/Codebase.md"

# Usage.md heading renames. Two reasons, both user-facing:
#   - a backticked stage identifier is noise in a *user guide* heading; it moves
#     into the first line of the section body, where it is more useful;
#   - GitHub drops `—` and `&` but keeps the spaces around them, leaving a
#     double hyphen in the anchor.
# Codebase.md is deliberately absent: it is a code reference, its headings ARE
# identifiers, and its slugs are invisible to a reader.
RENAMES = {
    "1. Concatenate (`concat`)": "1. Concatenate",
    "2. Axial strain (`strain`)": "2. Axial strain",
    "3. Mosaicity volume (`mosaicity`)": "3. Mosaicity volume",
    "4. Aligned rocking volumes (`rocking`)": "4. Aligned rocking volumes",
    "5. Visualize volumes (`visualize`)": "5. Visualize volumes",
    "6. ParaView export (`paraview`)": "6. ParaView export",
    "7. Oblique slices (`slices`)": "7. Oblique slices",
    "8. Line profiles (`profiles`)": "8. Line profiles",
    "9. Rocking-matched layers (`matched`)": "9. Rocking-matched layers",
    "Regions of interest — two windows, two frames": (
        "Regions of interest: two windows, two frames"
    ),
    'The "Publication style…" editor': "The Publication style editor",
    "Jobs from marks… (profiles)": "Jobs from marks (profiles)",
    "Tips & troubleshooting": "Tips and troubleshooting",
}

CALLOUTS = {
    "info": "NOTE",
    "example": "TIP",
    "note": "NOTE",
    "tip": "TIP",
    "important": "IMPORTANT",
    "warning": "WARNING",
    "caution": "CAUTION",
}

_FENCE = re.compile(r"^\s*(?:```|~~~)")
_HEADING_LINE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_WIKI = re.compile(r"\[\[([^\]|\n]+)(?:\|([^\]\n]+))?\]\]")


def slugify(heading: str) -> str:
    text = re.sub(r"<[^>]*>", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def heading_slugs(text: str) -> dict[str, str]:
    slugs: dict[str, str] = {}
    seen: dict[str, int] = {}
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        m = None if in_fence else _HEADING_LINE.match(line)
        if not m:
            continue
        heading = m.group(2)
        base = slugify(heading)
        n = seen.get(base, 0)
        seen[base] = n + 1
        slugs[heading] = base if n == 0 else f"{base}-{n}"
    return slugs


def map_lines(text: str, fn):
    """Apply *fn* to every line outside a fenced code block."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        out.append(line if in_fence else fn(line))
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def rename_headings(text: str) -> str:
    def fix(line: str) -> str:
        m = _HEADING_LINE.match(line)
        if m and m.group(2) in RENAMES:
            return f"{m.group(1)} {RENAMES[m.group(2)]}"
        return line

    return map_lines(text, fix)


def collapse_to_details(text: str) -> str:
    """`> [!note]- Summary` + its quoted body -> a real <details> block.

    GitHub has no collapsed-callout syntax; <details>/<summary> is the native
    one, and Obsidian renders it too, so this is a gain in both.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = re.match(r"^>[ \t]*\[![A-Za-z]+\]-[ \t]*(.*)$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        summary = m.group(1).strip() or "Details"
        i += 1
        body: list[str] = []
        while i < len(lines) and lines[i].startswith(">"):
            line = lines[i]
            body.append(line[2:] if line.startswith("> ") else line[1:])
            i += 1
        while body and not body[-1].strip():
            body.pop()
        out += ["<details>", f"<summary><b>{summary}</b></summary>", ""]
        out += body
        out += ["", "</details>"]
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def normalise_callouts(text: str) -> str:
    def fix(line: str) -> str:
        m = re.match(r"^(>[ \t]*)\[!([A-Za-z]+)\]([-+]?)(.*)$", line)
        if not m:
            return line
        kind = CALLOUTS.get(m.group(2).lower())
        if kind is None:
            raise SystemExit(f"unmapped callout type {m.group(2)!r}: {line}")
        return f"{m.group(1)}[!{kind}]{m.group(3)}{m.group(4)}"

    return map_lines(text, fix)


def rewrite_wiki_links(text: str, own: dict[str, str], usage: dict[str, str]) -> str:
    def one(m: re.Match) -> str:
        target, alias = m.group(1), m.group(2)
        if target.startswith("#"):
            head = RENAMES.get(target[1:], target[1:])
            if head not in own:
                raise SystemExit(f"no heading for {target!r} in this document")
            return f"[{alias or head}](#{own[head]})"
        if "#" in target:
            doc, head = target.split("#", 1)
            if doc != "Usage":
                raise SystemExit(f"unexpected cross-doc wiki link {target!r}")
            head = RENAMES.get(head, head)
            if head not in usage:
                raise SystemExit(f"no heading for {head!r} in docs/Usage.md")
            return f"[{alias or head}](Usage.md#{usage[head]})"
        return f"[{alias or target + '.md'}]({target}.md)"

    return map_lines(text, lambda line: _WIKI.sub(one, line))


def main() -> int:
    usage = rename_headings(USAGE.read_text(encoding="utf-8"))
    codebase = CODEBASE.read_text(encoding="utf-8")

    usage_slugs = heading_slugs(usage)
    codebase_slugs = heading_slugs(codebase)

    usage = rewrite_wiki_links(usage, usage_slugs, usage_slugs)
    codebase = rewrite_wiki_links(codebase, codebase_slugs, usage_slugs)

    usage = collapse_to_details(usage)
    usage = normalise_callouts(usage)
    codebase = normalise_callouts(codebase)

    USAGE.write_text(usage, encoding="utf-8")
    CODEBASE.write_text(codebase, encoding="utf-8")
    print("swept docs/Usage.md and docs/Codebase.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the sweep**

```bash
cd /home/albert/Desktop/darq_xray && python3 <scratchpad>/sweep_docs.py
```

Expected: `swept docs/Usage.md and docs/Codebase.md`, exit 0. Any `SystemExit`
message means the script hit a link it could not resolve — every target is
known to resolve today, so that is a script bug. Fix the script and re-run from
a clean `git checkout -- docs/`; never hand-patch around it.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python3 -m pytest -q tests/test_docs_links.py tests/test_docs_dependencies.py tests/test_docs_stage_params.py
```

Expected: all PASS. `test_docs_stage_params` passing is the check that the
`<details>` conversion did not eat any parameter table.

- [ ] **Step 6: Review the diff by hand**

```bash
git diff --stat docs/
git diff docs/Usage.md | head -400
```

Confirm specifically:
- the 9 `<details>` blocks each have a blank line after `</summary>` and before
  `</details>` (without it the tables inside do not render), and their
  parameter tables are intact;
- no line inside a fenced code block changed;
- `[[Usage#8. Line profiles (\`profiles\`)|profiles]]` in `docs/Codebase.md`
  became `[profiles](Usage.md#8-line-profiles)` — the *renamed* anchor;
- the mermaid diagrams in both files are untouched.

- [ ] **Step 7: Commit**

```bash
git add tests/test_docs_links.py docs/Usage.md docs/Codebase.md
git commit -F - <<'EOF'
docs: render on GitHub, and pin it with a test

The two guides were written in an Obsidian vault and used its dialect.
GitHub renders [[wiki links]] as literal text outside a GitHub *wiki*, so
the entire Contents block of the user guide arrived on the public
repository as unclickable noise and 92 cross-references went nowhere;
[!info] and [!example] callouts showed their own markup.

Converts every wiki link to [text](#github-slug), maps the two
Obsidian-only callout types onto the five GitHub renders (Obsidian
renders those too, so nothing is lost there), and turns the nine
collapsed "Advanced fields" callouts into <details> blocks, which are
native on GitHub and work in Obsidian as well.

Usage.md's stage headings drop their backticked identifiers: in a *user*
guide the identifier belongs in the body line, not the heading. Codebase.md
keeps its heading text untouched — it is a code reference, its headings
are identifiers, and its slugs are invisible to a reader.

tests/test_docs_links.py is the guard, for the same reason
test_docs_dependencies.py and test_docs_stage_params.py exist: a
documentation contract that nothing checks rots quietly, and this one
demonstrably did.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WQFycbBb3cQzEnBAfTmYVH
EOF
```

---

## Task 2: Rewrite `README.md` as a landing page

**Files:**
- Modify (full rewrite): `README.md`
- Create: `docs/images/README.md`

**Interfaces:**
- Consumes: nothing from Task 1 except that `docs/Usage.md`'s anchors are now
  GitHub slugs, so the README may deep-link into it, e.g.
  `[the stage reference](docs/Usage.md#stage-reference)`.
- Produces: the canonical wording of the install block and the nine-stage
  table, which Task 3 refers to rather than duplicates.

### Required structure

Write the file in exactly this order. Content requirements per section are
below; write the prose, do not copy these bullets as prose.

1. `# DARQ` — then **one sentence with no acronym in it** saying what it does
   (a desktop app that turns X-ray microscopy data into strain and mosaicity
   maps), then two lines unpacking DFXM and ESRF ID03.

2. **Screenshot slot.** Immediately after the intro, an HTML comment:

   ```html
   <!-- Add a screenshot: save the main window as docs/images/main-window.png,
        then delete this comment and uncomment the line below.
   ![The DARQ main window](docs/images/main-window.png)
   -->
   ```

   Nothing must render as a broken image before the PNG exists.

3. `## Is this for you?` — a short list. Must name: BLISS `.h5` scans from ESRF
   ID03; that you run **darfix** yourself and it produces `maps.h5`; that the
   outputs are strain maps, mosaicity maps, aligned 3-D volumes, ParaView
   exports, oblique slices and line profiles.

4. `## What you need first` — Python 3.10 or newer; a desktop with a display
   (it is a GUI, not a batch tool — though every stage also has a CLI); darfix
   installed separately; `ffmpeg` on `PATH` optional, enabling MP4 export with
   a GIF fallback without it.

5. `## Install` — the Global Constraints install block verbatim, then:

   ```markdown
   ## Launch it

   ```bash
   darq_xray
   ```
   ```

   Then **one** `<details>` block titled something like *If the install goes
   wrong, or you want to know why these steps* containing three short
   subsections:
   - `error: externally-managed-environment` — PEP 668; Debian/Ubuntu's system
     Python refuses to install into itself and suggests
     `--break-system-packages`. Make the venv instead.
   - Why `-e` is not optional — `experiments/` sits **beside** the package, not
     inside it, so it is not installed as package data; a plain
     `pip install .` starts the app with an empty preset dropdown and no
     calibration.
   - Why a venv at all — keeps a heavy Qt/VTK stack off the system Python, and
     is what CI does.

6. `## Your first run` — six numbered steps against the real UI:
   1. app opens on the **Overview** page, pipeline drawn left to right;
   2. pick an **experiment preset** (`STO2_overnight` ships with it) and point
      its `raw_root` / `processed_root` at your own data — the shipped paths
      are `/path/to/data/ESRF/…` placeholders;
   3. run **Concatenate** if your scans are not already concatenated (it is
      optional);
   4. run **darfix yourself, outside this app** — it produces `maps.h5`;
   5. back in the app, run **Axial strain**;
   6. what you get: `stacked_strain_volumes.h5` plus a folder of PNGs, and a
      green banner. Name the artifact so the reader can tell it worked.

7. `## What the nine stages do` — the table, **two columns only** (`Stage`,
   `What it does`), plain English, with darfix as its own row marked
   *external — you run this yourself*. Do **not** carry over the
   `Ported from` column. Link the stage names into
   `docs/Usage.md#stage-reference` where useful.

8. `## Before you trust a strain map` — a `> [!WARNING]` callout: the reference
   angle (`ccmth reference`) and the pixel scales are physical calibration for
   *your* beamtime; wrong values produce meaningless maps; the app flags those
   fields with a red **⚠ calibration** marker; confirm them against the
   beamline calibration.

9. `## Documentation` — links to `docs/Usage.md` (user guide),
   `docs/Codebase.md` (code reference), `CLAUDE.md` (contributor conventions).
   One line each on who it is for.

10. `## Development` — `pip install -e ".[test]"`, `python3 -m pytest -q`,
    `ruff check . && ruff format .`, CI runs both on every push, and the
    docs-in-the-same-change contract with its two enforcing tests named. Then
    the `<!-- deps:start -->` block (see Global Constraints — the ten names,
    unchanged), then a `<details>` holding the dependency-span discussion moved
    down from the old README: only lower bounds are declared; CI pins the floor
    on one leg (numpy 1.26.4 / scipy 1.11.4 / matplotlib 3.6.3) and takes the
    newest on the other; an older Python alone does **not** select older
    dependencies; `np.percentile` on float32 returns float64 on numpy 1.x and
    float32 on 2.x; `VERIFIED_SCIPY_VERSIONS`, `slices.RSS_FLOOR_BYTES` and
    `volumeio.centring_scaffold_bytes`; `tests/conftest.py` isolates
    `QSettings`.

11. `## License` — MIT, link `LICENSE`, and keep the existing PySide6 LGPLv3
    note about redistributing a bundled binary.

### Deletions (deliberate — do not reinstate)

- The **Origin** blockquote about the `Scripts2` scripts. Provenance a user
  never needs, explaining a self-skipping test a fresh clone cannot observe. It
  survives in `CLAUDE.md`'s own *Provenance* section and in `docs/Codebase.md`.
- The `Ported from` column. `y_calc_axial_strain_v6_batch` means nothing to
  anyone outside the original project.
- The `## Layout` directory tree. That is `docs/Codebase.md`'s job; replace it
  with at most three lines of prose in the Development section (Qt-free core in
  `darq_xray/`, the PySide6 app in `darq_xray/gui/`, one module per stage in
  `darq_xray/stages/`).
- `## Domain constraints carried into the core`. Physics detail belonging to
  the user guide; only the calibration warning survives, as section 8.

- [ ] **Step 1: Read the current README before editing it**

```bash
cat README.md
```

Needed to lift the exact wording of the pieces that survive — the LGPL note,
the psutil/dependency facts, the shipped-preset placeholder-paths fact.

- [ ] **Step 2: Create the screenshot directory and its note**

Create `docs/images/README.md`:

```markdown
# Screenshots

Drop the files below in here, then uncomment the matching `![...]` line in
`README.md` (and in `docs/Usage.md` for the first one).

| File | What to capture |
|---|---|
| `main-window.png` | The whole window on the **Overview** page — the pipeline rail, the experiment header, the stage chips. |
| `stage-panel.png` | A stage mid-run: the parameter form, the cost line under the Run button, the Log tab. |
| `figure-builder.png` | A finished strain map, or a composed multi-panel figure from the figure builder. |

Use the app's **light** theme — a screenshot on a white background reads on
both GitHub themes. PNG, and crop to the window rather than the whole desktop.
```

- [ ] **Step 3: Write the new `README.md`**

Replace the file wholesale with the structure above.

- [ ] **Step 4: Verify the docs tests still pass**

```bash
python3 -m pytest -q tests/test_docs_dependencies.py tests/test_docs_links.py
```

Expected: PASS. `test_docs_dependencies` is the one that catches a lost
`deps` block, so run it before anything else.

- [ ] **Step 5: Check every relative link resolves on disk**

```bash
grep -o '](\([^)#]*\)[^)]*)' README.md | sed 's/^](//; s/)$//; s/#.*//' \
  | grep -v '^https\?:' | sort -u | while read -r p; do
      [ -z "$p" ] || { [ -e "$p" ] && echo "OK   $p" || echo "MISS $p"; }
    done
```

Expected: every line starts `OK`. A `MISS` means a link to a file that does not
exist — `test_docs_links.py` checks anchors, not plain file links, so this one
is manual.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/images/README.md
git commit -F - <<'EOF'
docs(readme): rewrite the landing page for someone who just arrived

The README was ordered for a contributor and for the historical record:
it reached the Scripts2 provenance and a ~200-word CI dependency-span
policy before it reached "how do I install this". A beamline scientist
with DFXM data and no Python background got no answer to either of their
two questions — does this handle my data, and how do I run it.

Reordered so those two come first: what it is in one acronym-free
sentence, who it is for, what you need, then five install lines with no
extras bracket and therefore no shell-quoting rule to explain. The venv
is `venv`, not the hidden `.venv`. `pip install -e ".[test]"` moves to
Development, where the test extra is actually wanted — it adds pytest and
pytest-timeout and nothing a user running the app needs.

The three "why" explanations (PEP 668, why editable, why a venv) move
into a collapsed block: present when the install breaks, invisible when
it does not. The dependency-span discussion moves to Development for the
same reason.

Drops the Origin blockquote, the `Ported from` column and the directory
tree; provenance stays in CLAUDE.md and Codebase.md, the layout in
Codebase.md. Adds docs/images/ with the screenshots to capture — the
image lines are written but commented out, so nothing renders broken
before the PNGs exist.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WQFycbBb3cQzEnBAfTmYVH
EOF
```

---

## Task 3: Rewrite the front of `docs/Usage.md`

**Files:**
- Modify: `docs/Usage.md` — the YAML frontmatter through the end of the
  `## Quick start` section (currently lines 1–103, ending at the `---` rule
  before `## Core concepts`). **Everything from `## Core concepts` down is not
  touched by this task.**

**Interfaces:**
- Consumes: the renamed headings and GitHub-slug anchors from Task 1; the
  install block and stage table wording from Task 2.
- Produces: the new heading set that Task 1's `test_docs_links.py` will
  validate, and the section openers Task 4 extends.

### Required structure

```
---            (YAML frontmatter — KEPT, unchanged)
# DARQ — Usage Guide
> [!NOTE] What this is                     (reworded, acronyms defined)
> [!NOTE] This document is kept in sync…   (kept, retargeted anchor)
## Contents                                (markdown links — see below)
## What this app does
## Before you start
## Install and launch
## Words you'll meet
## Your first analysis, end to end
---
## Core concepts                           (Task 4 onward; untouched here)
```

**`## Contents`** — plain markdown links, one per top-level section, in file
order. Every anchor must be the GitHub slug of a heading that exists after Task
1's renames:

```markdown
- [What this app does](#what-this-app-does)
- [Before you start](#before-you-start)
- [Install and launch](#install-and-launch)
- [Words you'll meet](#words-youll-meet)
- [Your first analysis, end to end](#your-first-analysis-end-to-end)
- [Core concepts](#core-concepts)
  - [Regions of interest: two windows, two frames](#regions-of-interest-two-windows-two-frames)
- [The pipeline at a glance](#the-pipeline-at-a-glance)
- [Stage reference](#stage-reference)
- [Publication export](#publication-export)
- [Figure builder](#figure-builder)
- [Interactive viewers](#interactive-viewers)
- [Tips and troubleshooting](#tips-and-troubleshooting)
- [Running without the GUI (CLI)](#running-without-the-gui-cli)
- [Maintaining this guide](#maintaining-this-guide)
```

Note `#words-youll-meet` — the apostrophe is dropped by the slug rule, not
turned into a hyphen.

**`## What this app does`** — one paragraph. Define DFXM and ID03 on first use.
Say plainly that it is nine stages driven from one window, and that darfix sits
in the middle and is not part of this app.

**`## Before you start`** — what you need on the machine (Python 3.10+, a
display, darfix, optional ffmpeg) and what data you need (BLISS `.h5` scans
and/or a darfix `maps.h5`). Then a short "where darfix fits" paragraph. This
fact is currently only discoverable as a `[!note]` under a mermaid diagram
halfway down the file; state it before anything asks the reader to run a stage.

**`## Install and launch`** — the Global Constraints install block, then
`darq_xray` to launch, then one line pointing at the README's collapsed
troubleshooting for the *why*. Do not re-explain PEP 668 here. Then:
- the `<!-- deps:start -->` block, unchanged (Global Constraints);
- the `> [!WARNING]` about not hand-picking packages off that list, retyped
  from the existing one at lines 75–80 — the psutil / "8 GB free on a machine
  with 467 GB free" incident, kept verbatim in substance;
- the `> [!WARNING]` that the install must stay editable.

**`## Words you'll meet`** — a two-column glossary table, `Term | What it
means`. One row each, one or two sentences, no forward references:

| Term | Must convey |
|---|---|
| DFXM | Dark-Field X-ray Microscopy; imaging a crystal grain through a diffracted beam |
| ID03 | The ESRF beamline this pipeline's file layout and motor names come from |
| darfix | The separate ESRF tool that fits a rocking curve per pixel; **you run it yourself**, between concat and the map stages |
| `maps.h5` | darfix's output — the per-pixel fit results the map stages read |
| rocking curve | Intensity vs. crystal angle at one pixel, measured by rotating the sample |
| COM / FWHM | Centre of mass and full width at half maximum of that curve — the two things darfix fits |
| ccmth | The monochromator angle motor; its reference value sets the strain zero, and it is calibration |
| axial strain | Relative lattice-spacing change along the scattering vector, from the COM angle via the cot method |
| mosaicity | Local crystal misorientation — how much the lattice tilts from place to place |
| samy / samz | The sample translation motors; `samy` steps across the beam and its shift is what the alignment corrects |
| layer | One `samz` position — a single 2-D slice of the sample |
| stack / volume | Layers assembled into a 3-D array, all on one common grid |
| ROI | Region of interest. There are **two** here, in two frames — link to `#regions-of-interest-two-windows-two-frames` |
| VDS | HDF5 Virtual DataSet — a file that *references* other files instead of copying them |
| PVTI | The ParaView image-data format the `paraview` stage writes |
| preset / experiment | A named YAML file in `experiments/` holding the paths, calibration and ROIs shared by every stage |
| stage | One step of the pipeline: a form, a Run button, and files out |

**`## Your first analysis, end to end`** — a numbered walkthrough on
`STO2_overnight`. Must name, concretely: the preset dropdown and *Edit…*; that
the shipped `raw_root` / `processed_root` are `/path/to/data/ESRF/…`
placeholders to repoint (or use *Initialize from data…*); running concat;
leaving the app to run darfix; coming back for strain; and the artifacts that
appear (`stacked_strain_volumes.h5`, the PNG folder, the green banner). Keep
the existing `> [!WARNING]` about calibration values being physical, at the end
of this section.

Also add the screenshot slot as an HTML comment near the top, matching Task 2's
form, referencing `images/main-window.png` (relative to `docs/`).

- [ ] **Step 1: Read the region being replaced**

```bash
python3 - <<'PY'
import pathlib
print("".join(pathlib.Path("docs/Usage.md").read_text().splitlines(keepends=True)[:110]))
PY
```

Read it once, in full, before editing — the surviving callouts must be retyped
from the real bytes, not from memory (they contain em-dashes and `…`).

- [ ] **Step 2: Replace lines 1–103 with the new front matter**

Everything from `## Core concepts` down stays exactly as Task 1 left it.

- [ ] **Step 3: Verify the docs tests pass**

```bash
python3 -m pytest -q tests/test_docs_links.py tests/test_docs_dependencies.py tests/test_docs_stage_params.py
```

Expected: all PASS. `test_docs_links` proves every Contents anchor resolves —
this is the exact bug being fixed, so a failure here means a slug was
mispredicted; fix the link, not the test.

- [ ] **Step 4: Confirm no reference-body content was lost**

```bash
git diff --stat docs/Usage.md
grep -n '^## ' docs/Usage.md
```

Expected: the diff touches only the top of the file, and the section list runs
`What this app does` → `Before you start` → `Install and launch` →
`Words you'll meet` → `Your first analysis, end to end` → `Core concepts` →
`The pipeline at a glance` → `Stage reference` → … unchanged from there.

- [ ] **Step 5: Commit**

```bash
git add docs/Usage.md
git commit -F - <<'EOF'
docs(usage): open the guide with what a newcomer needs

The guide used roughly fifteen domain terms — DFXM, darfix, maps.h5,
rocking curve, COM/FWHM, ccmth, mosaicity, axial strain, samy/samz,
layer, stack, VDS, PVTI, ROI, preset — and defined none of them, so a
reader who had the data but not the project could not get past the first
screen. It also duplicated the README's install prose at length, and put
"darfix runs outside this app" in a note under a diagram halfway down,
after several sections had already told the reader to run a stage.

Replaces the front matter with: what this app does, before you start
(including where darfix fits, stated up front), a short install that
defers to the README for the why, a glossary of every term the rest of
the guide assumes, and an end-to-end first analysis on the shipped
STO2_overnight preset naming the artifacts that prove it worked.

The Contents block becomes real markdown links. Everything from Core
concepts down is unchanged — the per-parameter reference is good, it was
just unreachable for anyone who did not already know the vocabulary.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WQFycbBb3cQzEnBAfTmYVH
EOF
```

---

## Task 4: Plain-language openers, and final verification

The reference body assumes context in the same way the front matter did, one
section at a time. This task adds a lead sentence to each major section — what
this stage or subsystem is *for*, in the vocabulary of the glossary — without
touching the detail beneath it.

**Files:**
- Modify: `docs/Usage.md` — one added opening sentence (occasionally two) per
  section listed below. Nothing is deleted.

**Interfaces:**
- Consumes: the glossary from Task 3 (openers may use its terms freely, and
  should link into it where a term is load-bearing).
- Produces: nothing later tasks depend on.

### Where an opener is required

For each stage section under `## Stage reference`, the first line of the body
becomes the identifier line, then the opener:

```markdown
### 7. Oblique slices

*Stage id `slices` · headless: `python3 -m darq_xray.stages.slices --help`*

Cuts flat planes at any angle through the aligned volumes, so you can look
at a feature that does not lie along X, Y or Z. …
```

The identifier line replaces what the heading used to carry, and is required
for all nine stages: `concat`, `strain`, `mosaicity`, `rocking`, `visualize`,
`paraview`, `slices`, `profiles`, `matched`.

An opener is also required at the top of:
- `## Core concepts` — one sentence saying these apply to every stage.
- `## The pipeline at a glance` — one sentence before the mermaid diagram
  saying how to read it.
- `## Publication export` — what it is *for* (one session-global style applied
  to every figure a stage writes).
- `## Figure builder` — what it is for (composing finished maps and traces into
  one multi-panel figure at a fixed µm-per-cm scale).
- `## Interactive viewers` — one sentence saying these open on demand and build
  nothing until clicked.
- `## Running without the GUI (CLI)` — one sentence on when you would want this
  (batch, scripting, a headless machine).

Keep each opener to one or two sentences. Do not restate what the section
already says further down; do not add new facts that are not already true
elsewhere in the docs.

- [ ] **Step 1: Add the nine stage identifier lines and openers**

Read each stage section's existing first paragraph before writing its opener —
the opener must not contradict it. Use `Read` with `offset`/`limit` on the
section rather than re-reading the whole file.

- [ ] **Step 2: Add the six remaining section openers**

- [ ] **Step 3: Verify the docs tests pass**

```bash
python3 -m pytest -q tests/test_docs_links.py tests/test_docs_dependencies.py tests/test_docs_stage_params.py
```

Expected: all PASS.

- [ ] **Step 4: Run the full suite**

```bash
python3 -m pytest -q --ignore=tests/test_gui_viewer3d.py
```

Expected: the counts match the pre-change baseline plus the new
`test_docs_links.py` cases. The baseline on this machine is **1882 passed, 13
skipped** on Python 3.12. `tests/test_gui_viewer3d.py` cannot run in this
environment and `tests/gui_smoke.py` segfaults at step [24] on clean master;
report both **NOT RUN** rather than claiming them green.

- [ ] **Step 5: Lint**

```bash
ruff check . && ruff format --check .
```

Expected: clean. `docs/` is excluded from ruff, so this only covers the new
test file.

- [ ] **Step 6: Commit**

```bash
git add docs/Usage.md
git commit -F - <<'EOF'
docs(usage): say what each section is for before saying how it works

Every stage section opened straight into its inputs and outputs, which
answers "how" for someone who already knows why they are here and nothing
at all for someone deciding whether this stage is the one they want. Adds
one or two plain sentences at the top of each stage and each major
subsystem, in the vocabulary the new glossary establishes.

The stage identifier and its headless CLI move into a line under each
heading — the place the backticked identifier went when the headings were
cleaned for GitHub anchors, and more useful there than in a heading.

Nothing is deleted; the per-parameter detail is unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WQFycbBb3cQzEnBAfTmYVH
EOF
```

---

## Final state

After Task 4 the branch `docs/user-facing-overhaul` holds five commits: the
spec, then one per task. Still open, deliberately, and **not** part of this
plan:

- **Screenshots.** `docs/images/` exists with instructions; the three PNGs do
  not. The `![...]` lines are written but commented out in `README.md` and
  `docs/Usage.md`. Adding a screenshot is then a one-line uncomment.
- **The GitHub render check.** The only verification that actually proves the
  stated goal is looking at the two files on github.com after pushing:
  Contents clickable, callouts coloured, `<details>` blocks expanding. Do this
  before merging to `master`, and remember that pushing to `master` publishes
  immediately.
- **Merging.** Use the `finish-and-record` skill. Check `git remote -v` first
  — the remote exists (`origin`, public), so unlike earlier work in this repo
  the push step is real and irreversible.
