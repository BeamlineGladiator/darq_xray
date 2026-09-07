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

Occurrences inside fenced code blocks and inline `code spans` are exempt
throughout: prose that *shows* `[[wiki link]]` syntax is documentation, and
backticked text is never a link in the first place.
"""

from __future__ import annotations

import os.path
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "docs/Usage.md", "docs/Codebase.md")

#: The only alert types GitHub renders as a coloured callout. Obsidian renders
#: all five as well, so this set is the intersection, not a compromise.
GITHUB_ALERT_TYPES = frozenset({"NOTE", "TIP", "IMPORTANT", "WARNING", "CAUTION"})

_FENCE = re.compile(r"^\s*(?:```|~~~)")
#: An inline code span. Longest-run-first so ``double`` backticks win.
_CODE_SPAN = re.compile(r"``[^`\n]+``|`[^`\n]+`")
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_WIKI = re.compile(r"\[\[[^\]\n]+\]\]")
_CALLOUT = re.compile(r"^>\s*\[!([A-Za-z]+)\][-+]?", re.MULTILINE)
#: `](#anchor)` and `](Other.md#anchor)`; the doc part may be empty. The
#: character class excludes ":" and so never matches an external URL fragment.
_ANCHOR = re.compile(r"\]\(([A-Za-z0-9_./-]*)#([^)\s]+)\)")
#: `![alt](src)`. The alt text may wrap across lines, so the class must not
#: exclude newlines — it is prose, and these docs wrap at ~78 columns.
_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def slugify(heading: str) -> str:
    r"""GitHub's heading -> anchor rule (`github-slugger`).

    Lowercase, drop HTML tags, drop every character that is not a word
    character, a space or a hyphen, then spaces to hyphens. ``\w`` is used with
    its Unicode meaning so a non-ASCII heading keeps its letters; what this
    removes here in practice is backticks, dots, parentheses, colons, commas,
    quotes, ``&``, ``…`` and the em-dash.
    """
    text = re.sub(r"<[^>]*>", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


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


def _prose(text: str) -> str:
    """*text* with fenced blocks and inline code spans emptied.

    What survives is the part a reader clicks. `README.md` legitimately names
    the `[[wiki link]]` syntax in prose, and a link's own text may be
    backticked (``[`alignment.py`](#alignmentpy)``) — emptying the span leaves
    the link's target intact either way.
    """
    return _CODE_SPAN.sub("", _without_fences(text))


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


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def unresolved_anchors(rel_path: str) -> list[str]:
    """Anchors in *rel_path* that name no heading in their target document."""
    body = _prose(_read(rel_path))
    own_slugs = set(heading_slugs(_read(rel_path)).values())
    bad: list[str] = []
    for doc, anchor in _ANCHOR.findall(body):
        if not doc:
            valid = own_slugs
            where = rel_path
        else:
            target = doc if doc.endswith(".md") else f"{doc}.md"
            # Always relative to the *linking document's* directory, so both
            # `docs/Usage.md` from the README and `../README.md` from docs/
            # resolve. Anything else silently misjudges one of the two.
            where = os.path.normpath(str(Path(rel_path).parent / target))
            if not (ROOT / where).exists():
                bad.append(f"#{anchor} -> missing document {where}")
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
def test_every_embedded_image_exists(rel_path):
    """A doc must not embed a screenshot that is not in the repository.

    The screenshots arrived after the prose that references them, written first
    as commented-out placeholders. A stale reference is the natural failure of
    that order, and on GitHub it renders as a broken-image icon on the landing
    page — the most visible possible place for it.
    """
    missing = []
    for src in _IMAGE.findall(_prose(_read(rel_path))):
        if src.startswith(("http://", "https://", "data:")):
            continue
        where = os.path.normpath(str(Path(rel_path).parent / src))
        if not (ROOT / where).exists():
            missing.append(f"{src} -> {where}")
    assert not missing, (
        f"{rel_path} embeds {len(missing)} image(s) that do not exist:\n  "
        + "\n  ".join(missing)
        + "\n\nAdd the file, or remove the reference — GitHub renders a missing "
        "image as a broken-image icon, it does not hide it."
    )


@pytest.mark.parametrize("rel_path", DOCS)
def test_no_obsidian_wiki_links(rel_path):
    found = _WIKI.findall(_prose(_read(rel_path)))
    assert not found, (
        f"{rel_path} still contains {len(found)} Obsidian wiki link(s), which "
        f"GitHub renders as literal text: {sorted(set(found))[:10]}\n"
        "Use [text](#github-slug), or [text](Other.md#github-slug) across docs."
    )


@pytest.mark.parametrize("rel_path", DOCS)
def test_callouts_use_types_github_renders(rel_path):
    used = {t.upper() for t in _CALLOUT.findall(_prose(_read(rel_path)))}
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
    # GitHub drops "&" but keeps the spaces either side: a double hyphen.
    assert slugify("Tips & troubleshooting") == "tips--troubleshooting"

    # Two headings with *identical* text collapse to one map entry, keeping the
    # later slug. That is a limitation, not the case worth pinning; the case
    # worth pinning is distinct headings colliding on one slug.
    colliding = "# A B\n\ntext\n\n## A-B\n"
    assert sorted(heading_slugs(colliding).values()) == ["a-b", "a-b-1"]

    assert _WIKI.findall("see [[#Somewhere]]") == ["[[#Somewhere]]"]
    # A fence that *shows* the syntax is documentation, not a broken link.
    assert not _WIKI.findall(_prose("```\n[[#Somewhere]]\n```"))
    # Prose that *names* the syntax is documentation, not a link.
    assert not _WIKI.findall(_prose("we used `[[wiki links]]` before"))
    # A `# comment` in a bash fence is not a heading.
    assert heading_slugs("```bash\n# not a heading\n```\n\n## Real\n") == {"Real": "real"}
    assert _CALLOUT.findall("> [!info] x") == ["info"]
    # Alt text wraps across lines in these docs; the pattern must survive it.
    assert _IMAGE.findall("![a very long\nalt text](images/x.png)") == ["images/x.png"]
