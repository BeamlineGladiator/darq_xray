---
name: finish-and-record
description: Use when a feature branch in darq_xray is complete and the user says "wrap up", "merge it", "finish this", or picks a finish-branch option
---

# Finish and Record

Composes verify-suite → superpowers:finishing-a-development-branch →
handoff-note with this repo's facts. Run every step below; a wrap-up that
skips one is incomplete.

## Steps

1. **Verify the branch.** Run the `verify-suite` skill on the feature branch.
   Red → abort the finish; report and stop.
2. **Finish the branch** via superpowers:finishing-a-development-branch, with
   these repo facts pre-loaded:
   - **A git remote exists** (`origin`) — a push publishes the branch, so
     push or open a PR only when the user picks that option.
   - Merge convention: `--no-ff` merge commit (or ff-only when master ==
     merge-base).
   - Present the finish menu as AskUserQuestion numbered choices (CLAUDE.md).
3. **Re-run `verify-suite` on merged master** — a merge can break a branch
   that was green on its own. Cite the new master HEAD in the canonical report
   line.
4. **Cleanup.** Remove the worktree, `git worktree prune`, delete the branch;
   confirm `git status --short` on master is clean.
5. **Record.** Invoke the `handoff-note` skill. The note MUST contain, besides
   what changed:
   - **Deferred/next workstreams** — items explicitly not done and what was
     agreed to come next.
   - The **rollback hash** (pre-merge master HEAD).
   Memory files under `~/.claude/projects/.../memory/` are not git-tracked —
   the Write IS the save; never try to git-commit them.

## Failure handling

- Merge conflict → stop and report; never resolve silently.
- Post-merge verify red → offer revert to the rollback hash as a numbered
  AskUserQuestion option alongside fix-forward.
