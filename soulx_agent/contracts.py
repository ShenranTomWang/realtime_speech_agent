"""Small contracts shared by the orchestration, harness and inference layers."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelConnection:
    model: str
    base_url: str
    api_key_env: str


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    call: Callable[[dict], Awaitable[object]]


class AgentHarness(Protocol):
    async def run(self, prompt: str) -> str: ...


HarnessFactory = Callable[[str, str, list[Tool]], AgentHarness]
