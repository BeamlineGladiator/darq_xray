"""The dependency lists in the docs must match `pyproject.toml`.

`pyproject.toml` is the single source of truth: it is what `pip install -e .`
actually reads. The prose lists in `README.md` and `docs/Usage.md` exist for a
reader skimming the repo on GitHub, and a prose list that nothing checks drifts
— `docs/Usage.md` had been missing **pyyaml and psutil** since the lists were
written, and a psutil-less environment is exactly what produced the "8 GB free
of 502 GB RAM" bug fixed in `darq_xray/common/machine.py`.

Each doc marks its list with `<!-- deps:start -->` / `<!-- deps:end -->` so this
test extracts exactly the intended span rather than guessing at prose. HTML
comments are invisible in rendered Markdown (Obsidian and GitHub alike).
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS_WITH_LISTS = ("README.md", "docs/Usage.md")

_START, _END = "<!-- deps:start -->", "<!-- deps:end -->"


def _pyproject_dependency_names() -> set[str]:
    """Distribution names from `[project] dependencies`, lowercased."""
    tomllib = pytest.importorskip("tomllib")  # stdlib from 3.11; project allows 3.10
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    specs = data["project"]["dependencies"]
    # "PySide6>=6.5" -> "pyside6"; comparison is case-insensitive because the
    # docs spell PySide6 and pyyaml the way their projects do.
    return {re.split(r"[<>=!~ \[]", spec, maxsplit=1)[0].strip().lower() for spec in specs}


def _documented_names(rel_path: str) -> set[str]:
    text = (ROOT / rel_path).read_text(encoding="utf-8")
    assert _START in text and _END in text, (
        f"{rel_path} has no {_START} / {_END} markers around its dependency list"
    )
    block = text.split(_START, 1)[1].split(_END, 1)[0]
    names = {name.strip().lower() for name in re.findall(r"`([^`]+)`", block)}
    assert names, f"{rel_path}'s dependency block lists no `backticked` package names"
    return names


@pytest.mark.parametrize("rel_path", DOCS_WITH_LISTS)
def test_documented_dependencies_match_pyproject(rel_path):
    expected = _pyproject_dependency_names()
    documented = _documented_names(rel_path)
    assert documented == expected, (
        f"{rel_path} lists {sorted(documented)} but pyproject.toml declares "
        f"{sorted(expected)}; missing {sorted(expected - documented)}, "
        f"extra {sorted(documented - expected)}"
    )


def test_pyproject_is_installable():
    """`pip install -e .` needs both a build backend and explicit packages.

    Without `[build-system]` pip falls back to setuptools' auto-discovery,
    which refuses this flat layout outright ("Multiple top-level packages
    discovered in a flat-layout: ['darq_xray', 'experiments', 'tests']") — so
    the documented install command fails on a fresh clone. Both halves are
    asserted because either one alone leaves it broken.
    """
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data.get("build-system", {}).get("build-backend"), (
        "pyproject.toml has no [build-system]; `pip install -e .` cannot build it"
    )
    include = data["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "darq_xray*" in include, (
        f"package discovery must name the source tree explicitly, got {include}"
    )


def test_every_real_package_is_covered_by_discovery():
    """The `darq_xray*` glob must actually cover every importable package.

    Guards the failure this cannot otherwise catch: a new top-level package
    (or a stage tree moved out from under `darq_xray/`) silently missing from an
    installed copy while the run-in-place tests all still pass.

    `tests` is a package (it has an `__init__.py`, so `tests.peak_rss` and
    `tests.qt_helpers` import as modules) but is deliberately NOT shipped —
    excluded here by name rather than by widening the globs, so adding a
    *different* top-level package still fails this.
    """
    not_shipped = {"tests"}

    def _is_ours(path) -> bool:
        """Reject anything under a checkout directory that is not source.

        `pathlib.Path.glob` matches dotted names — unlike `glob.glob`, which
        skips them — so `*/**/__init__.py` walks straight into `.venv`. README
        and docs/Usage both tell the user to create exactly that, inside the
        clone, so without this filter the test fails for anyone who follows the
        install instructions: a real venv contributes hundreds of names like
        `.venv.lib.python3.12.site-packages.numpy`. `site-packages` is matched
        as well as the leading dot, because a venv is not required to be hidden
        or to be called `.venv`.
        """
        parts = path.relative_to(ROOT).parts
        return not any(
            part.startswith(".") or part in {"__pycache__", "site-packages", "node_modules"}
            for part in parts
        )

    found = {
        str(p.parent.relative_to(ROOT)).replace("/", ".")
        for p in ROOT.glob("*/**/__init__.py")
        if _is_ours(p)
    } | {
        str(p.parent.relative_to(ROOT)).replace("/", ".")
        for p in ROOT.glob("*/__init__.py")
        if _is_ours(p)
    }
    assert "darq_xray.stages" in found, "package scan found nothing — the glob is broken"
    uncovered = sorted(
        n for n in found if not n.startswith("darq_xray") and n.split(".")[0] not in not_shipped
    )
    assert not uncovered, (
        f"importable packages outside the darq_xray* glob: {uncovered} — "
        "add them to [tool.setuptools.packages.find] include, or they vanish "
        "from an installed copy"
    )
