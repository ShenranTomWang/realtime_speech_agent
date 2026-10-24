import asyncio
from pathlib import Path

import pytest

from soulx_agent.contracts import Tool
from soulx_agent.frontend_tools import load_override
from soulx_agent.orchestration import Coordinator


async def lookup(args):
    return {"found": True}


@pytest.mark.parametrize("names", [None, [], ["submit_task"], ["submit_task", "get_weather"]])
async def test_exact_frontend_allowlist_and_backend_isolation(tmp_path, names):
    calls = []

    class Harness:
        def __init__(self, role, prompt, tools):
            self.role = role
            calls.append((role, {tool.name: tool for tool in tools}))

        async def run(self, prompt):
            return "done"

    override = None
    if names is not None:
        override = tmp_path / "override.yaml"
        import yaml

        override.write_text(yaml.safe_dump({"tools": names}))
    domain = Tool("lookup", "domain lookup", {"type": "object"}, lookup)
    coordinator = Coordinator("policy", [domain], tmp_path / "events.jsonl", Harness, override)
    await coordinator.turn("hello")
    front = calls[0][1]
    if names is None:
        assert {"lookup", "submit_task", "task_status", "cancel_task"} <= front.keys()
    else:
        assert set(front) == set(names)
    # Exercise the real delegation path even if this frontend cannot invoke it.
    await coordinator.task_tools()[0].call({"request": "lookup"})
    await asyncio.wait_for(coordinator.delivery.get(), 1)
    assert calls[1][0] == "backend"
    assert set(calls[1][1]) == {"lookup"}
    await coordinator.close()


@pytest.mark.parametrize(
    "contents", ["null", "tools: all", "tools: [x, x]", "tools: []\nextra: true"]
)
def test_invalid_override_fails(tmp_path, contents):
    path = tmp_path / "override.yaml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_override(path)


def test_unknown_tools_fail_closed(tmp_path):
    path = tmp_path / "override.yaml"
    path.write_text("tools: [missing]")
    with pytest.raises(ValueError, match="Unknown frontend tools"):
        Coordinator("", [], tmp_path / "events.jsonl", None, path)


def test_experiment_profiles_have_exact_tools():
    root = Path(__file__).resolve().parents[1] / "config/frontend_tools"
    assert load_override(root / "delegate_only.yaml")["tools"] == ["submit_task"]
    assert load_override(root / "delegate_simple.yaml")["tools"] == [
        "submit_task",
        "get_current_datetime",
        "get_weather",
    ]
