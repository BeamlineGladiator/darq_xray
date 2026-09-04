---
name: finish-and-record
description: Use when a feature branch in darq_xray is complete and the user says "wrap up", "merge it", "finish this", or picks a finish-branch option
---

# Finish and Record

Composes verify-suite → superpowers:finishing-a-development-branch →
handoff-note with the repo facts and the two steps past sessions forgot
(post-merge re-verify; deferred-work enumeration). Every step below is
REQUIRED; a wrap-up that skips one is incomplete.

## Steps

1. **Verify the branch.** Run the `verify-suite` skill on the feature branch.
   Red → abort the finish; report and stop.
2. **Finish the branch** via superpowers:finishing-a-development-branch, with
   these repo facts pre-loaded:
   - **No git remote exists** (`git remote -v` is empty) — skip pull/push/PR
     options entirely; the merge menu is local-only.
   - Merge convention: `--no-ff` merge commit (or ff-only when master ==
     merge-base).
   - Present the finish menu as AskUserQuestion numbered choices (CLAUDE.md).
3. **Re-run `verify-suite` on merged master.** This is not optional — the
   post-merge re-verify caught real breakage in past sessions. Cite the new
   master HEAD in the canonical report line.
4. **Cleanup.** Remove the worktree, `git worktree prune`, delete the branch;
   confirm `git status --short` on master is clean.
5. **Record.** Invoke the `handoff-note` skill. The note MUST contain, besides
   what changed:
   - **Deferred/next workstreams** — items explicitly not done and what was
     agreed to come next (a past handoff omitted this and the user had to
     probe for it).
   - The **rollback hash** (pre-merge master HEAD).
   Memory files under `~/.claude/projects/.../memory/` are not git-tracked —
   the Write IS the save; never try to git-commit them.

## Failure handling

- Merge conflict → stop and report; never resolve silently.
- Post-merge verify red → offer revert to the rollback hash as a numbered
  AskUserQuestion option alongside fix-forward.
