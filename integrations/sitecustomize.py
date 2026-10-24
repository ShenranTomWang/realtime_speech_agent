"""Opt-in bootstrap, including benchmark subprocesses. Never enabled by default."""

import os

benchmark = os.environ.get("SOULX_BENCHMARK")
if benchmark:
    try:
        if benchmark == "tau_voice":
            from soulx_agent.benchmarks.tau_voice import install
        elif benchmark == "eva":
            from soulx_agent.benchmarks.eva import install
        else:
            raise ValueError(f"Unknown benchmark adapter: {benchmark}")
        install()
    except Exception:
        # Python otherwise swallows sitecustomize errors and could run the wrong model.
        import traceback

        traceback.print_exc()
        os._exit(78)
