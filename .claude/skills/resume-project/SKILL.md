---
name: resume-project
description: Use when the user says "resume", "resume last session", "continue the plan", "pick up where we left off", or any terse continuation prompt at session start in dfxm_pipeline
---

# Resume Project

Safely re-enter a multi-session project from its handoff note. A bare "resume"
authorizes *scoping*, not execution.

**Never auto-launch a no-check-in skill (subagent-driven-development,
executing-plans) from a bare resume.** That exact misfire happened: agents were
dispatched within seconds and the user had to hard-interrupt. The SDD "execute
all tasks without stopping" rule applies *after* scope is confirmed — not to
scoping itself.

## Steps

1. **Load state, Read-first.** Read `MEMORY.md` (it is injected into context at
   session start, but the Edit tool still requires a real Read — 6 past
   sessions hit "File has not been read yet" on it). Read the relevant project
   note. For an SDD/worktree flow, also Read `.superpowers/sdd/progress.md`
   and the plan file **once, now** — later, slice the plan with
   `offset`/`limit` per task; never re-read it in full per cycle (9 full
   re-reads wasted in one session).
2. **Verify reality matches the note.** Worktree exists? Branch name?
   `git log --oneline -3` vs the note's recorded HEAD/rollback point?
   `git status --short` clean? Any mismatch → stop and report the discrepancy
   before executing anything.
3. **Honor standing preferences from memory** without being re-told
   (currently: mid-tier subagents on the `sonnet-4-6` agent type, never
   Sonnet 5).
4. **State the resume point and the single next action** — e.g. "Task 3
   implemented at d48e8ff but unreviewed; next: dispatch its review."
5. **Ask scope via AskUserQuestion** (numbered, per CLAUDE.md):
   (1) run autonomously to completion, (2) next task/phase then stop,
   (3) just report state. Include the agreed stop point in the option text so
   the user never has to hard-interrupt at a boundary.

## Failure handling

- Handoff note missing → say so; offer to reconstruct state from
  `git log` + the progress ledger.
- Note stale (HEAD moved past its recorded state) → treat the note's "next
  action" as suspect; re-derive from the ledger and flag the drift.

## Red flags

- Dispatching any agent before the AskUserQuestion in step 5 has been answered.
- Editing MEMORY.md or a progress ledger you haven't Read this session.
- "The user said resume, so they want full autonomy" — they want *continuity*;
  scope is a separate question.
