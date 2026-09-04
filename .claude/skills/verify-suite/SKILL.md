---
name: verify-suite
description: Use when about to claim the darq_xray suite is green — before any commit, merge, "done"/"passing" claim, fresh-worktree baseline check, or plan phase boundary; also when the user asks "run the suite" or "is it green?"
---

# Verify Suite

Canonical green-check for this repo. A report is valid only if all four checks
ran to completion with observed exit codes — never infer success from partial
output, and never report green with a check skipped.

## Steps

1. **Provenance first.** Echo `pwd`, `git rev-parse --abbrev-ref HEAD`,
   `git rev-parse --short HEAD`, `git status --short`. If the tree is dirty,
   say so in the report. Every report cites HEAD + branch so handoff notes and
   memory entries are unambiguous.
2. **Run the four checks in order**, capturing each exit code:

   | Check | Command | Parse |
   |---|---|---|
   | tests | `python3 -m pytest -q` | final `N passed, M skipped` line (report actual counts, don't assume) |
   | lint | `ruff check .` | exit 0 |
   | format | `ruff format --check .` | exit 0 |
   | GUI smoke | `python3 tests/gui_smoke.py` (with a timeout) | final `[N]/[N]` step count |

   The smoke test is `tests/gui_smoke.py` — no `test_` prefix, not a pytest
   file; it sets `QT_QPA_PLATFORM=offscreen` itself.
3. **Report one canonical line:**
   `HEAD <hash> on <branch>: pytest N passed/M skipped, ruff clean, format clean, gui_smoke X/X`

## Failure handling

- Any non-zero exit → STOP. No commit, merge, or "done" claim. Report which
  check failed with its tail output. Distinguish "check failed" from "check
  couldn't run" (missing dep, wrong directory) and say which.
- gui_smoke hang or core dump (exit 144): capture output, then kill by
  **stored PID** — run it via `python3 tests/gui_smoke.py & PID=$!` and
  `kill $PID`. Never `pkill -f gui_smoke` or `pkill -f spawn_main`: the
  pattern matches your own shell (6 self-kills in one past session). Before
  deep Qt debugging, inspect the smoke test itself for variable shadowing —
  that was the actual root cause last time.

## Red flags — the report is invalid if you catch yourself thinking

- "pytest passed, the rest surely does too" — run all four.
- "I ran it earlier this session" — HEAD moved, run again.
- "gui_smoke is slow, skip it this once" — then the report must say
  "gui_smoke: NOT RUN", not green.
