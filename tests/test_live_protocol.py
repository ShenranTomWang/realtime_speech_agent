"""Exercise the real OpenAI SDK against a live loopback gateway, without model APIs."""

import asyncio
import json
import socket

import httpx
import uvicorn
from openai import AsyncOpenAI

from soulx_agent import server
from soulx_agent.orchestration import Coordinator


async def test_openai_sdk_audio_and_tool_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SOULX_RUN_DIR", str(tmp_path))

    class Audio:
        def __init__(self, rate):
            pass

        async def connect(self):
            pass

        async def close(self):
            pass

    class Harness:
        def __init__(self, model, system_prompt, tools):
            self.tools = {item.name: item for item in tools}

        async def run(self, prompt):
            result = await self.tools["lookup"].call({"order_id": "ABC"})
            assert json.loads(result)["status"] == "shipped"
            return "Your order has shipped."

    class TestCoordinator(Coordinator):
        def __init__(self, *args, **kwargs):
            kwargs["harness"] = Harness
            super().__init__(*args, **kwargs)

    async def synthesize(text):
        return bytes(960)

    monkeypatch.setattr(server, "SoulXAudio", Audio)
    monkeypatch.setattr(server, "Coordinator", TestCoordinator)
    monkeypatch.setattr(server, "synthesize", synthesize)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    host = uvicorn.Server(uvicorn.Config(server.app, log_level="error", lifespan="on"))
    serving = asyncio.create_task(host.serve(sockets=[listener]))
    while not host.started:
        await asyncio.sleep(0.01)
    client = AsyncOpenAI(api_key="local", base_url=f"http://127.0.0.1:{port}/v1")
    session_id = None
    try:
        async with client.realtime.connect(model="soulx_claude_code") as conn:
            await conn.session.update(
                session={
                    "type": "realtime",
                    "tools": [
                        {
                            "type": "function",
                            "name": "lookup",
                            "description": "Look up an order",
                            "parameters": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                            },
                        }
                    ],
                }
            )
            await conn.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Track my order."}],
                }
            )
            await conn.response.create()
            audio_received = False
            tool_received = False
            async with asyncio.timeout(5):
                async for event in conn:
                    if event.type == "session.updated":
                        session_id = event.session.id
                    elif event.type == "response.function_call_arguments.done":
                        assert event.name == "lookup"
                        tool_received = True
                        await conn.conversation.item.create(
                            item={
                                "type": "function_call_output",
                                "call_id": event.call_id,
                                "output": '{"status":"shipped"}',
                            }
                        )
                        await conn.response.create()
                    elif event.type == "response.output_audio.delta":
                        audio_received = True
                        assert event.delta
                    elif event.type == "response.output_audio_transcript.done":
                        assert event.transcript == "Your order has shipped."
                    elif event.type == "response.done":
                        assert event.response.status == "completed"
                        break
                    elif event.type == "error":
                        raise AssertionError(event)
            assert audio_received and tool_received and session_id
        async with httpx.AsyncClient(trust_env=False) as http:
            status = await http.get(f"http://127.0.0.1:{port}/sessions/{session_id}")
            assert status.status_code == 200, (status.status_code, status.headers, status.text)
            assert not status.json()["error"]
            await http.delete(f"http://127.0.0.1:{port}/sessions/{session_id}")
    finally:
        await client.close()
        host.should_exit = True
        await asyncio.wait_for(serving, 5)
        listener.close()
