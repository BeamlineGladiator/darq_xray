"""Every declared stage parameter must appear in both user-facing docs.

CLAUDE.md makes "update `docs/Usage.md` and `docs/Codebase.md` in the SAME
change" a contract, but until this test nothing enforced it for *parameters* —
only dependencies were pinned (`test_docs_dependencies.py`). The contract was
therefore honour-system and rotted quietly: a sweep in 2026-09 found 19 params
missing from `Codebase.md` and 50 from `Usage.md`, several of them whole
control surfaces (every `paraview` export switch was undocumented, so nothing
told a user they could turn an export off, or that NaN replacement was on by
default).

A param counts as documented if the doc names it in any of three ways, because
the two docs legitimately write about parameters differently:

1. **By identifier** (`replace_nan`) — how `Codebase.md`, a code reference,
   normally names one.
2. **By GUI label** ("Replace NaN") — `Usage.md` is a user guide and may name
   the field the user actually sees rather than the identifier behind it.
3. **By a collective wildcard** (`volume_opacity_<key>`, `include_*`) — the
   `visualize` stage generates eighteen opacity params from one table, and both
   docs describe them as a family. Enumerating eighteen near-identical rows
   would be worse documentation, not better, so the family form counts for
   every member.

The test deliberately checks only that a parameter is *mentioned*. It cannot
judge whether the prose is any good — but "not mentioned anywhere" is exactly
the failure that actually happened, and it is cheap to catch.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("docs/Usage.md", "docs/Codebase.md")


def _specs():
    """Stage schemas, via the GUI bindings that own the canonical stage list."""
    pytest.importorskip("PySide6")
    from darq_xray.gui.bindings import STAGE_SPECS

    return STAGE_SPECS


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def is_documented(param, text: str) -> bool:
    """True when *text* names *param* by identifier, label, or family form."""
    if param.name in text:
        return True

    label = (param.label or "").strip()
    if label and _squash(label) in _squash(text):
        return True

    # A family form is any prefix of the identifier ending at an underscore and
    # followed by a placeholder: `volume_opacity_<key>` covers
    # `volume_opacity_strain`, `include_*` covers `include_raw_sum`.
    for cut in range(len(param.name) - 1, 3, -1):
        stem = param.name[:cut]
        if stem.endswith("_") and re.search(re.escape(stem) + r"(<[a-z_]+>|\*|\{)", text):
            return True
    return False


@pytest.mark.parametrize("rel_path", DOCS)
def test_every_stage_param_is_documented(rel_path):
    text = (ROOT / rel_path).read_text(encoding="utf-8")
    undocumented = [
        f"{stage}.{p.name} (label {p.label!r})"
        for stage, spec in _specs().items()
        for p in spec.params
        if not is_documented(p, text)
    ]
    assert not undocumented, (
        f"{len(undocumented)} stage parameter(s) appear nowhere in {rel_path} — "
        "by identifier, GUI label, or family form:\n  "
        + "\n  ".join(undocumented)
        + f"\n\nAdd them to {rel_path} in the same change that declared them "
        "(see CLAUDE.md > Documentation)."
    )


def test_the_check_can_actually_fail():
    """Guard the guard: a param named nowhere must be reported as missing.

    Without this, a bug in `is_documented` (an over-broad family regex, say)
    would make the test above pass vacuously and silently stop protecting the
    contract it exists to protect.
    """

    class _FakeParam:
        name = "zzz_param_that_is_not_in_any_doc"
        label = "Zzz Param That Is Not In Any Doc"

    for rel_path in DOCS:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert not is_documented(_FakeParam(), text), (
            f"is_documented() called an absent param documented in {rel_path}"
        )
