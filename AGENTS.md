Use four-space bracket wrapping and --arg_name Python CLI flags.
Keep SoulX as the audio/turn detector; never substitute transcript replay for voice evaluation.
Frontend and backend must both execute through Claude Agent SDK (Claude Code harness).
Only benchmark-owned tools may be exposed during evaluation. Never leak reference answers to agents.
Keep speech interruption separate from accepted task cancellation.
LLM hosting/routing belongs in llm_runner; the agent harness receives only ModelConnection.
Frontend/backend LLM runners must remain independently configurable for local or remote inference.
Read credentials only for selected runners, from environment variables. Do not log keys.
Slurm: source "${HOME_DIR}/.bashrc", then scratch environment; no set -euo pipefail,
no log redirection, no shared path redefinitions. Submit from each script's directory.
Do not submit inference/evaluation jobs until the user requests submission.

Debug interactively; do not add automated tests, startup smoke checks, or credential prevalidation.
