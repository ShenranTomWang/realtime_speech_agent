import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from soulx_agent import server
from soulx_agent.orchestration import Coordinator


class FakeHarness:
    started = None
    finish = None

    def __init__(self, model, prompt, tools):
        self.model = model
        self.tools = {item.name: item for item in tools}

    async def run(self, prompt):
        if self.model == "backend":
            self.started.set()
            await self.finish.wait()
            return "Your order is out for delivery."
        text = json.loads(prompt)["conversation"][-1]["content"]
        if text == "delegate":
            await self.tools["submit_task"].call({"request": "Track my order"})
            return "I will check."
        return "Hello."


@pytest.fixture
def harness():
    FakeHarness.started = asyncio.Event()
    FakeHarness.finish = asyncio.Event()
    return FakeHarness


async def test_backend_does_not_block_foreground(tmp_path, harness):
    coordinator = Coordinator("policy", [], tmp_path / "events.jsonl", harness)
    assert await coordinator.turn("delegate") == "I will check."
    await asyncio.wait_for(harness.started.wait(), 1)
    assert await asyncio.wait_for(coordinator.turn("hello"), 1) == "Hello."
    assert list(coordinator.tasks.values())[0]["status"] == "running"
    harness.finish.set()
    task_id, result = await asyncio.wait_for(coordinator.delivery.get(), 1)
    assert "delivery" in result
    assert coordinator.tasks[task_id]["status"] == "completed"
    await coordinator.close()


async def test_explicit_cancellation(tmp_path, harness):
    coordinator = Coordinator("policy", [], tmp_path / "events.jsonl", harness)
    await coordinator.turn("delegate")
    await harness.started.wait()
    task_id = next(iter(coordinator.tasks))
    cancel = next(tool for tool in coordinator.task_tools() if tool.name == "cancel_task")
    assert await cancel.call({"task_id": task_id}) == {"status": "cancelled"}
    await coordinator.close()
    assert coordinator.delivery.empty()


async def test_barge_in_keeps_backend_alive(tmp_path, monkeypatch, harness):
    monkeypatch.setenv("SOULX_RUN_DIR", str(tmp_path))
    events = []

    async def send(event):
        events.append(event)

    session = server.VoiceSession("unit", {}, send)
    session.coordinator.harness = harness
    await session.coordinator.turn("delegate")
    await harness.started.wait()
    session.playback = asyncio.create_task(asyncio.sleep(30))
    await session.interrupt()
    await asyncio.sleep(0)
    assert session.playback.cancelled()
    assert not next(iter(session.coordinator.tasks.values()))["future"].cancelled()
    harness.finish.set()
    await asyncio.wait_for(session.coordinator.delivery.get(), 1)
    assert events[-1]["type"] == "response.cancelled"
    await session.close()


async def test_remote_tool_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SOULX_RUN_DIR", str(tmp_path))
    events = asyncio.Queue()
    session = server.VoiceSession("tools", {}, events.put)
    task = asyncio.create_task(session.call_tool("lookup", {"id": "A"}))
    event = await events.get()
    assert event["name"] == "lookup"
    await session.handle(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": event["call_id"],
                "output": '{"found":true}',
            },
        }
    )
    assert await task == '{"found":true}'
    assert not session.pending_tools
    await session.close()


async def test_backend_delivery_waits_for_user(tmp_path, monkeypatch):
    monkeypatch.setenv("SOULX_RUN_DIR", str(tmp_path))

    async def send(event):
        pass

    session = server.VoiceSession("delivery", {}, send)
    spoken = asyncio.Event()

    async def speak(text, epoch=None):
        spoken.set()

    session.speak = speak
    session.user_speaking = True
    session.spawn(session.deliver())
    await session.replies.put(("task", "result", None))
    await asyncio.sleep(0.08)
    assert not spoken.is_set()
    session.user_speaking = False
    await asyncio.wait_for(spoken.wait(), 1)
    await session.close()


def test_realtime_handshake_and_audio_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SOULX_RUN_DIR", str(tmp_path))

    class Audio:
        def __init__(self, rate):
            pass

        async def connect(self):
            pass

        async def close(self):
            pass

        async def process(self, pcm):
            return {"state": "nonidle"}

    monkeypatch.setattr(server, "SoulXAudio", Audio)
    with TestClient(server.app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json({"type": "session.update", "session": {"tools": []}})
            session_id = ws.receive_json()["session"]["id"]
            ws.send_json({"type": "input_audio_buffer.append", "audio": "AAAAAA=="})
            event = ws.receive_json()
            assert event["type"] == "input_audio_buffer.speech_started"
        assert client.delete("/sessions/" + session_id).json()["closed"]
        assert client.get("/").status_code == 200


def test_reserved_tools_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("SOULX_RUN_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="reserved"):
        server.VoiceSession("reserved", {"tools": [{"name": "submit_task"}]}, None)
