CRITICAL: Never add fallback values or silent error handling. Causes masked failures and incorrect embedding results.
CRITICAL: Pure Python library only. No frontend, no servers, no web frameworks.
CRITICAL: Maximum brevity. No pleasantries. No explanations unless asked. Code and facts only.
CRITICAL: Zero commentary. Output only what was explicitly requested. No observations, no opinions, no unsolicited notes.
CRITICAL: Test errors must be resolved whether preexisting or not. Never skip or ignore failing tests.

# Brevity Examples
# BAD:  "I found the bug — it's on line 42 where the null check is missing. Want me to fix it?"
# GOOD: "Line 42: missing null check. Fix?"
# BAD:  "The feature is already implemented in chonk/community/_index.py. Want me to mark it complete?"
# GOOD: "Implemented at chonk/community/_index.py:64. Mark complete?"

# Agents & Skills
Agent definitions live in `.claude/agents/*.md`. Read the relevant agent file before acting in that role.
Skill definitions live in `.claude/skills/*/SKILL.md`. Read the relevant skill file before applying it.

# Requirements Tracking
On any new requirement, constraint, feature, or design decision: append it to `docs/arch/requirements.md` inline before proceeding. Do not skip this step. Silent — no commentary.

# Swarm Mode (Self-Claim)
All agents operate autonomously. No lead assignment required.

## After completing any task:
1. Check the task list — find lowest-ID task that is unblocked AND unowned
2. Claim it (set yourself as owner, status in_progress)
3. Begin work immediately

## Rules:
- Never wait for assignment — self-claim lowest unblocked/unowned task
- If no tasks available, report idle and stop
- Respect module boundaries — only claim tasks in your domain
- Run verification command for your module when done
- After completing a task, loop back to step 1 — always pull next task

## Spawning Teammates
Include: 1) Which module(s) they own 2) Files to read first 3) Verification command 4) What NOT to touch
