import json

from soulx_agent.audio import SoulXAudio
from soulx_agent.benchmarks.fdb_v3 import tool_definitions, upstream_policy
from soulx_agent.contracts import ModelConnection, Tool
from soulx_agent.harness import ClaudeHarness


async def test_soulx_float32_protocol():
    class Socket:
        async def send(self, raw):
            self.payload = json.loads(raw)

        async def recv(self):
            return '{"state":{"state":"speak","text":"hello"}}'

    audio = SoulXAudio(16000)
    audio.ws = Socket()
    state = await audio.process(bytes(3200))
    assert state["text"] == "hello"
    assert audio.ws.payload["type"] == "audio"
    import base64

    assert len(base64.b64decode(audio.ws.payload["audio"])) == 6400


async def test_sdk_options_and_mcp_dispatch(monkeypatch):
    import claude_agent_sdk as sdk

    captured = {}

    async def execute(args):
        return {"found": args["id"]}

    class Client:
        def __init__(self, options):
            captured["options"] = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def query(self, prompt):
            captured["prompt"] = prompt

        async def receive_response(self):
            yield sdk.ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="test",
                result="Done",
            )

    monkeypatch.setattr(sdk, "ClaudeSDKClient", Client)
    monkeypatch.setenv("SOULX_GATEWAY_KEY", "test-only")
    agent = ClaudeHarness(
        ModelConnection("backend", "http://127.0.0.1:4000", "SOULX_GATEWAY_KEY"),
        "policy",
        [
            Tool(
                "lookup",
                "Look up",
                {"type": "object", "properties": {"id": {"type": "string"}}},
                execute,
            )
        ],
    )
    assert await agent.run("request") == "Done"
    options = captured["options"]
    assert options.tools == []
    assert options.permission_mode == "dontAsk"
    assert options.allowed_tools == ["mcp__session__lookup"]
    assert options.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert options.setting_sources == []


def test_fdb_schemas_do_not_use_answers():
    class Registry:
        @staticmethod
        def lookup(order_id: str, **kwargs) -> dict:
            return {}

        FUNCTIONS = {"lookup": lookup}

    tools = tool_definitions(Registry)
    assert tools[0]["parameters"]["required"] == ["order_id"]
    assert set(tools[0]["parameters"]["properties"]) == {"order_id"}


def test_fdb_policy_extracts_only_agent_prompt(tmp_path):
    (tmp_path / "lk_agent_tool.py").write_text(
        'class VoiceAgent:\n    def __init__(self):\n        super().__init__(instructions="Use tools.")\n'
    )
    assert upstream_policy(tmp_path) == "Use tools."
