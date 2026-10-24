"""Worker fixture: exercise real tool RPC without loading audio or language models."""

from sglang_omni.scheduling.session import SessionScheduler

from soulx_agent.omni_stage import FrontendHooks
from soulx_agent.server import VoiceSession


class ToolSession(VoiceSession):
    async def start(self):
        pass

    async def handle(self, message):
        if message["type"] == "test.tool":

            async def invoke():
                result = await self.call_tool("lookup", {"value": 7})
                await self.emit("test.result", result=result)

            self.spawn(invoke())
        else:
            await super().handle(message)


class ToolHooks(FrontendHooks):
    session_type = ToolSession


def create_scheduler():
    return SessionScheduler(ToolHooks(), max_open_sessions=2, max_concurrency=2)
