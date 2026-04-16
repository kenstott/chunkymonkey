CRITICAL: Never add fallback values or silent error handling. Causes masked failures and incorrect embedding results.
CRITICAL: Pure Python library only. No frontend, no servers, no web frameworks.
CRITICAL: Maximum brevity. No pleasantries. No explanations unless asked. Code and facts only.
CRITICAL: Test errors must be resolved whether preexisting or not. Never skip or ignore failing tests.

# Requirements Tracking
On any new requirement, constraint, feature, or design decision: spawn a background haiku agent. It reads `.claude/agents/requirements-tracker.md` for format, then appends to `docs/arch/requirements.md`. Silent — skip implementation details, bugs, questions.

# Swarm Mode & Teammate Spawning
@.claude/refs/swarm-mode.md