"""Claude Code execution and MCP tools, independent of model hosting."""

import json
import os

from .contracts import ModelConnection, Tool


class ClaudeHarness:
    def __init__(self, connection: ModelConnection, system_prompt: str, tools: list[Tool]):
        self.connection = connection
        self.system_prompt = system_prompt
        self.tools = tools
        self.client = None
        self.steering = []
        self.accepting_steering = True

    async def steer(self, instruction: str) -> bool:
        if not self.accepting_steering:
            return False
        self.steering.append(instruction)
        if self.client is not None:
            await self.client.interrupt()
        return True

    async def run(self, prompt: str) -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
            create_sdk_mcp_server,
            tool,
        )

        sdk_tools = []
        for definition in self.tools:

            async def invoke(arguments, definition=definition):
                result = await definition.call(arguments)
                return {"content": [{"type": "text", "text": json.dumps(result)}]}

            sdk_tools.append(
                tool(definition.name, definition.description, definition.schema)(invoke)
            )
        server = create_sdk_mcp_server(name="session", version="1.0.0", tools=sdk_tools)
        token = os.environ[self.connection.api_key_env]
        options = ClaudeAgentOptions(
            model=self.connection.model,
            system_prompt=self.system_prompt,
            tools=[],
            mcp_servers={"session": server},
            allowed_tools=[f"mcp__session__{item.name}" for item in self.tools],
            permission_mode="dontAsk",
            setting_sources=[],
            strict_mcp_config=True,
            max_turns=32,
            effort=os.environ.get("SOULX_AGENT_EFFORT", "low"),
            env={
                "ANTHROPIC_BASE_URL": self.connection.base_url,
                "ANTHROPIC_AUTH_TOKEN": token,
                "ANTHROPIC_API_KEY": "",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            },
        )
        final = ""
        completed = False
        try:
            async with ClaudeSDKClient(options=options) as client:
                self.client = client
                while True:
                    if self.steering:
                        prompt += "\n" + json.dumps({"steering": self.steering})
                        self.steering.clear()
                    await client.query(prompt)
                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            final = " ".join(
                                block.text for block in message.content if isinstance(block, TextBlock)
                            )
                        elif isinstance(message, ResultMessage):
                            if self.steering:
                                break
                            self.accepting_steering = False
                            if message.is_error:
                                raise RuntimeError(
                                    f"Claude Code run failed: {message.subtype}: {message.result}"
                                )
                            final = message.result or final
                            completed = True
                    if completed or not self.steering:
                        break
                    prompt = (
                        "Continue the same task using the following updated instructions. "
                        "Retain prior context and completed tool results; do not repeat completed "
                        "side effects. Earlier side effects cannot be undone by this update."
                    )
                    final = ""
        finally:
            self.accepting_steering = False
            self.client = None
        if not completed:
            raise RuntimeError("Claude Code ended without a successful result")
        return final
