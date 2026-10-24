"""Real multiprocess Omni session graph, not a mock coordinator."""

import asyncio
import uuid
from pathlib import Path

import pytest
import yaml

pytest.importorskip("sglang_omni")

from soulx_agent.omni_runtime import OmniRuntime, OmniVoiceSession


async def test_omni_graph_tool_round_trip_and_detach(tmp_path, monkeypatch):
    monkeypatch.setenv("SOULX_RUN_DIR", str(tmp_path / "sessions"))
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/omni.yaml").read_text())
    config["stages"][0]["factory_path"] = "tests.omni_helpers.create_scheduler"
    config_path = tmp_path / "omni.yaml"
    config_path.write_text(yaml.safe_dump(config))
    runtime = OmniRuntime(config_path)
    events = asyncio.Queue()
    await runtime.start()
    session = OmniVoiceSession(runtime, uuid.uuid4().hex, {}, events.put)
    try:
        await session.start()
        assert (await session.status())["error"] is None
        await session.handle({"type": "test.tool"})
        event = await asyncio.wait_for(events.get(), timeout=10)
        assert event["type"] == "response.function_call_arguments.done"
        assert event["name"] == "lookup"
        session.detach()
        session.attached.set()
        await session.handle(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": event["call_id"],
                    "output": "confirmed",
                },
            }
        )
        result = await asyncio.wait_for(events.get(), timeout=10)
        assert result == {**result, "type": "test.result", "result": "confirmed"}
        assert (await session.status())["session_id"] == session.id
    finally:
        await session.close()
        await runtime.close()
