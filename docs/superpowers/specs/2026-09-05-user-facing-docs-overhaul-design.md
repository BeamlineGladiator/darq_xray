---
title: User-facing documentation overhaul — design
tags: [darq, documentation, readme, usage, github, design]
---

# User-facing documentation overhaul — design

**Date:** 2026-09-05
**Status:** approved, not yet implemented
**Touches:** `README.md`, `docs/Usage.md`, `docs/Codebase.md` (compatibility only),
`tests/test_docs_links.py` (new), `docs/images/` (new)
**Does not touch:** `pyproject.toml`, any code under `darq_xray/`

## Problem

The repository is now public. Its user-facing surfaces were written by and for
someone who already knows the project, and three concrete failures follow from
that:

1. **The GitHub landing page does not serve a newcomer.** `README.md` reaches
   the provenance of the code (an *Origin* blockquote about the `Scripts2`
   scripts) before it reaches "how do I install this", and spends a
   ~200-word paragraph on the CI dependency-span policy above the *Running*
   section. Neither belongs in a first read.
2. **The install command is unnecessarily hostile.** `pip install -e ".[test]"`
   makes a first-time user parse an extras bracket, a shell-quoting rule that
   exists only for zsh, and a `test` extra they do not need — the extra adds
   `pytest` and `pytest-timeout` and nothing else. The venv is called `.venv`,
   a hidden directory, which a reader unfamiliar with Python tooling cannot see
   in a file manager or a plain `ls`.
3. **Obsidian-only Markdown breaks on GitHub.** `[[wiki links]]` render as
   literal text outside a GitHub *wiki*, so the entire Contents block of
   `docs/Usage.md` renders as unclickable noise, and 92 cross-references across
   the two docs go nowhere.

Underneath all three: the guide uses roughly fifteen domain terms — DFXM,
darfix, `maps.h5`, rocking curve, COM/FWHM, ccmth, mosaicity, axial strain,
samy/samz, layer, stack, VDS, PVTI, ROI, preset — without defining any of
them.

## Audiences

Every decision below is justified against one of these four.

| Reader | Arrives at | Needs first |
|---|---|---|
| **Beamline scientist / PhD student with DFXM data** (primary) | README, then Usage | "Does this handle *my* data, and how do I run it?" |
| **Evaluator** — colleague, reviewer, recruiter | README only | What it is, a picture, whether it is real |
| **Contributor / future maintainer** | README → `CLAUDE.md` | Tests, lint, CI, the docs contract |
| **Reader of a paper** | README | Provenance, license, how to cite |

The current README is ordered for #4 and #3. The rewrite orders it for #1 and
#2 and moves #3's and #4's material down, without deleting it.

## Decisions

Recorded because each was a live choice, and the reasoning is what a later
session will need.

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Scope | README + `Usage.md` rewritten; `Codebase.md` gets the mechanical compatibility pass only | Rewriting `Codebase.md`'s prose: it is a code reference for an already-technical audience, 2160 lines, and its density is appropriate there |
| `Usage.md` treatment | Rewrite the front matter, preserve the reference body verbatim in substance | A separate `Getting-started.md`: a third doc to keep in sync, and `test_docs_stage_params.py` would then need both files scanned |
| Headline install | `python3 -m venv venv` + `pip install -e .` | Keeping `".[test]"`: makes a newcomer parse extras syntax and shell quoting to install a test runner they will not use |
| Link style | GitHub-first: `[text](#github-slug)` | `[[#Heading]]`: invisible to the primary audience. A dual TOC: two lists to keep in sync, exactly the drift the existing docs tests exist to prevent |
| Callouts | The five GitHub types, uppercase | Dropping callouts entirely: the calibration hazards genuinely warrant a coloured box, and the five types render in Obsidian too |
| `pyproject.toml` | Unchanged | Adding an `all`/`dev` extra alias, or an `install.sh`: more surface to maintain, and a shell script hides the tooling from the reader learning it |

**Accepted cost of the link decision.** Obsidian resolves `[[#Heading]]`
reliably and a `#github-slug` anchor less so, so in-Obsidian clicking may
degrade. GitHub correctness wins because that is where the primary audience
reads. Callouts, YAML frontmatter and mermaid all keep working in both tools.

## Design

### 1. `README.md` — a landing page

Rewritten top to bottom in this order, jargon-last:

1. **Title and one plain sentence** with no acronym in it, then two lines that
   unpack DFXM and ID03.
2. **Screenshot** — `docs/images/main-window.png`. See §5.
3. **Is this for you?** — a short list naming the data (BLISS `.h5` scans from
   ESRF ID03, a darfix `maps.h5`) so the wrong reader can leave in ten seconds.
4. **What you need first** — Python 3.10 or newer, a desktop with a display,
   darfix installed separately, `ffmpeg` optional (MP4 export; GIF without it).
5. **Install** — five lines, no brackets and therefore no quoting to explain:

   ```bash
   git clone https://github.com/BeamlineGladiator/darq_xray.git
   cd darq_xray
   python3 -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -e .
   ```

   Followed by **Launch it**: `darq_xray`.

   The three "why" explanations — PEP 668 / `externally-managed-environment`,
   why the install must stay editable (`experiments/` is not package data, so a
   non-editable install starts with an empty preset dropdown), and why a venv
   at all — go into **one collapsed `<details>` block** underneath: present
   when the install breaks, invisible when it does not.
6. **Your first run** — six numbered steps against the real UI, ending at a
   named artifact (`stacked_strain_volumes.h5` plus a folder of PNGs) so the
   reader can tell whether it worked.
7. **What the nine stages do** — the existing table reduced to *Stage* and
   *What it does*, in plain English, with darfix as an explained row rather
   than a footnote.
8. **Before you trust a strain map** — a `> [!WARNING]` box: the reference
   angles and pixel scales are physical calibration; wrong values produce
   meaningless maps.
9. **Documentation** — links to `docs/Usage.md`, `docs/Codebase.md`,
   `CLAUDE.md`.
10. **Development** — `pip install -e ".[test]"`, pytest, ruff, CI, the
    docs-in-the-same-change contract, and the entire dependency-span
    discussion, most of it inside a `<details>`.
11. **License and citation.**

**Deleted from the README:**

- the *Origin* blockquote about `Scripts2` — provenance a user never needs, and
  it explains a self-skipping test that a fresh clone cannot even observe;
- the `Ported from` column of the stage table — `y_calc_axial_strain_v6_batch`
  means nothing to anyone outside the original project;
- the `Layout` directory tree — that is `Codebase.md`'s job; three lines of
  prose replace it;
- *Domain constraints carried into the core* — physics detail that belongs in
  `Usage.md`; only the calibration warning survives, as item 8.

Provenance is not lost: it stays in `CLAUDE.md` (which already carries a
*Provenance* section) and in `Codebase.md`.

**Must be preserved:** the `<!-- deps:start -->` / `<!-- deps:end -->` block,
relocated into the Development section. `tests/test_docs_dependencies.py`
fails without it.

### 2. `docs/Usage.md` — new front, reference body intact

Everything from `## Stage reference` (currently line 689) downward keeps its
substance; it changes only by the mechanical sweep in §3 and by gaining a
plain-language opening sentence on each major section.

Above it, the front matter is replaced by:

| Section | Purpose |
|---|---|
| **What this app does** | One paragraph; every acronym defined on first use. |
| **Before you start** | What you need, what data, and **where darfix fits**. That last fact is currently only discoverable as a note under a mermaid diagram halfway down the file. |
| **Install and launch** | Short, deferring to the README. Replaces the ~40-line block that today duplicates the README's install prose. |
| **Words you'll meet** | **New glossary table.** DFXM, darfix, `maps.h5`, rocking curve, COM, FWHM, ccmth, mosaicity, axial strain, samy/samz, layer, stack, VDS, PVTI, ROI, preset, stage. The single largest comprehension unlock available: the guide uses all of these and defines none. |
| **Your first analysis, end to end** | A walkthrough on the shipped `STO2_overnight` preset: what to click, what appears, which file lands where. Replaces the four-bullet "Typical first run". |

The `<!-- deps:start -->` block stays in this file too (the test checks both
`README.md` and `docs/Usage.md`), inside *Install and launch*.

The existing `> [!WARNING]` about not hand-picking packages off the dependency
list — the one that records the psutil-less "8 GB free of 502 GB" bug — is
kept verbatim. It is a real incident and it is why the list is machine-checked.

### 3. GitHub compatibility sweep — all three docs

| Problem | Where | Fix |
|---|---|---|
| `[[wiki links]]` render as literal text | 49 in `Usage.md`, 43 in `Codebase.md` | `[text](#github-slug)`, or plain prose where the link was mid-sentence noise |
| `[!info]`, `[!example]` are Obsidian-only | 12 in `Usage.md`, 1 in `Codebase.md` | `[!NOTE]` and `[!TIP]` respectively; every callout uppercased |
| `> [!note]-` collapsed callouts | 9 in `Usage.md` | `<details><summary>…</summary>` — a real collapsible on GitHub *and* in Obsidian |
| Headings whose slugs read badly | ~12 in `Usage.md` | See the rename table below |

`Codebase.md` receives **only** this table. No prose rewrite.

**Heading renames in `Usage.md`** (the stage identifier is not lost — it moves
into the first line of the section body, as `Stage id: \`concat\` · CLI:
\`python3 -m darq_xray.stages.concat\``, which is more useful there than in a
heading):

| From | To | Resulting slug |
|---|---|---|
| `### 1. Concatenate (\`concat\`)` | `### 1. Concatenate` | `#1-concatenate` |
| `### 2. Axial strain (\`strain\`)` | `### 2. Axial strain` | `#2-axial-strain` |
| `### 3. Mosaicity volume (\`mosaicity\`)` | `### 3. Mosaicity volume` | `#3-mosaicity-volume` |
| `### 4. Aligned rocking volumes (\`rocking\`)` | `### 4. Aligned rocking volumes` | `#4-aligned-rocking-volumes` |
| `### 5. Visualize volumes (\`visualize\`)` | `### 5. Visualize volumes` | `#5-visualize-volumes` |
| `### 6. ParaView export (\`paraview\`)` | `### 6. ParaView export` | `#6-paraview-export` |
| `### 7. Oblique slices (\`slices\`)` | `### 7. Oblique slices` | `#7-oblique-slices` |
| `### 8. Line profiles (\`profiles\`)` | `### 8. Line profiles` | `#8-line-profiles` |
| `### 9. Rocking-matched layers (\`matched\`)` | `### 9. Rocking-matched layers` | `#9-rocking-matched-layers` |
| `### Regions of interest — two windows, two frames` | `### Regions of interest: two windows, two frames` | `#regions-of-interest-two-windows-two-frames` |
| `### The "Publication style…" editor` | `### The Publication style editor` | `#the-publication-style-editor` |
| `### Jobs from marks… (profiles)` | `### Jobs from marks (profiles)` | `#jobs-from-marks-profiles` |

The em-dash form is renamed because GitHub drops the `—` but keeps the spaces
either side, producing a double hyphen (`regions-of-interest--two-windows…`).

`[[Codebase]]` becomes `[Codebase.md](Codebase.md)`; `[[Usage]]` in
`Codebase.md` becomes `[Usage.md](Usage.md)`.

The 12 anchors already written as `](#…)` in `Usage.md` are correct GitHub
slugs, so this style is not new to the file.

**YAML frontmatter is kept** in both docs. GitHub renders it as a small table
above the title, which is tolerable; Obsidian needs it for `aliases`.

**Method.** The wiki-link and callout conversions are done by a throwaway
script in the scratchpad, not by hand, so the transformation is uniform and the
result is reviewable as a `git diff`. Every heading rename and every anchor is
then verified by the test in §4 rather than by eye.

### 4. `tests/test_docs_links.py` — new

Three assertions over `README.md`, `docs/Usage.md`, `docs/Codebase.md`:

1. **Every in-document `](#anchor)` resolves** to a heading in that same file,
   under GitHub's slug rules: lowercase; strip HTML tags; remove every
   character that is not a letter, digit, space, hyphen or underscore; replace
   spaces with hyphens; append `-1`, `-2`, … to a repeated slug. (No duplicate
   heading text exists in any of the three files today, but the dedupe rule is
   implemented so a future duplicate is handled rather than silently
   mismatched.)
2. **No `[[wiki link]]` survives** in the three shipped docs — excluding
   occurrences inside backticks, since the README legitimately mentions the
   syntax in prose.
3. **No callout uses a type GitHub does not render** — only `NOTE`, `TIP`,
   `IMPORTANT`, `WARNING`, `CAUTION`.

Plus a guard-the-guard test, matching the pattern
`test_docs_stage_params.py::test_the_check_can_actually_fail` already
establishes: a fabricated broken anchor must be reported.

Rationale: this is the same reasoning as the two existing docs tests. The
prose contract rotted quietly until something checked it; the link contract has
now demonstrably done the same.

### 5. `docs/images/` — new, one slot

The repository contains no screenshots. For a desktop application's landing
page that is the largest single readability gap, and it cannot be filled from
this session — there is no display here.

`docs/images/` is created with a short `README.md` naming exactly what to
capture:

1. `main-window.png` — the app on the Overview page, whole window;
2. `stage-panel.png` — a stage mid-run: form, cost line, log;
3. `figure-builder.png` — a finished strain map or a composed figure.

In `README.md` and `docs/Usage.md` the image line is written but left inside an
HTML comment, with a one-line instruction to uncomment it. **Nothing renders as
a broken image before the PNG exists**, and adding the screenshot is then a
one-line edit rather than a writing task.

`.gitignore` needs no change — it ignores `*.mp4` and `*.gif` but not `*.png`.

## Out of scope

- Any change to `pyproject.toml`, including new extras.
- An `install.sh` or `Makefile`.
- Prose changes to `docs/Codebase.md` beyond the §3 sweep.
- `CLAUDE.md` — it is a contributor file and is already fit for that audience.
- Publishing, tagging, or a release. This is a documentation change only.

## Verification

```bash
python3 -m pytest -q tests/test_docs_dependencies.py tests/test_docs_stage_params.py tests/test_docs_links.py
python3 -m pytest -q --ignore=tests/test_gui_viewer3d.py    # full suite
ruff check . && ruff format --check .
```

The full suite is required, not just the three docs tests:
`test_docs_stage_params.py` imports `darq_xray.gui.bindings`, so a docs change
can fail on a GUI import.

Expected: unchanged pass counts against the current baseline (1882 passed / 13
skipped on Python 3.12), plus the new `test_docs_links.py` cases.
`tests/test_gui_viewer3d.py` cannot run in this environment and
`tests/gui_smoke.py` segfaults at step [24] on clean master; both are reported
NOT RUN, per `CLAUDE.md`.

Manual check, and the only one that matters for the stated goal: **render both
files on GitHub after pushing** and confirm the Contents block is clickable,
every callout is a coloured box, and every collapsed section expands.

## Risks

| Risk | Mitigation |
|---|---|
| A parameter mentioned only in deleted README prose disappears from a doc | `test_docs_stage_params.py` checks `Usage.md` and `Codebase.md`, not the README, so README deletions cannot break it. The `Usage.md` reference body is preserved in substance for the same reason. |
| The `deps` block is lost in the README rewrite | `test_docs_dependencies.py` fails loudly on both files. Run it first, before the full suite. |
| The mechanical sweep mangles a link inside a code fence | The script skips fenced blocks; the `git diff` is reviewed before commit. |
| Heading renames break the 12 existing `](#…)` anchors | None of the 12 point at a renamed heading (they target `#replotting-…`, `#busy-indication`, `#pinned-planes-fast-re-runs`), and §4's test catches it if that is wrong. |
