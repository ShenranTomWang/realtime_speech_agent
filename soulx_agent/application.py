"""Wire the execution harness to role aliases exported by the LLM runner."""

import os

from .contracts import ModelConnection, Tool
from .harness import ClaudeHarness


def create_harness(role: str, system_prompt: str, tools: list[Tool]):
    connection = ModelConnection(
        model=role,
        base_url=os.environ.get("SOULX_GATEWAY_URL", "http://127.0.0.1:4000"),
        api_key_env="SOULX_GATEWAY_KEY",
    )
    return ClaudeHarness(connection, system_prompt, tools)
