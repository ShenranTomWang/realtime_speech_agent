"""SGLang-Omni session stage for the existing speech services and agent harnesses.

This CPU stage owns persistent conversation state. GPU model execution stays in
independent services; the stage never imports an LLM inference engine.
"""

import asyncio
import threading

from sglang_omni.proto.session import TimedChunk
from sglang_omni.scheduling.session import SessionHooks, SessionScheduler

from .server import VoiceSession

EVENT_FORMAT = "soulx_event"


class FrontendHooks(SessionHooks):
    session_type = VoiceSession

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.sessions = {}
        self.emitters = {}

    def call(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout=60)
        except BaseException:
            future.cancel()
            raise

    def open(self, session_identity, request):
        async def create():
            params = request.params

            async def send(event):
                self.emitters[session_identity](
                    TimedChunk("control", 0, 0, 0, event, format=EVENT_FORMAT)
                )

            self.sessions[session_identity] = self.session_type(
                params["session_id"], params["config"], send, web=params["web"]
            )

        self.call(create())

    def attach_output_stream(self, session_identity, emit):
        self.emitters[session_identity] = emit
        self.call(self.sessions[session_identity].start())

    def append(self, chunk, payload, context):
        async def handle():
            session = self.sessions[context.session_identity]
            message = chunk.payload
            if message["type"] == "soulx.status":
                await session.emit(
                    "soulx.status",
                    request_id=message["request_id"],
                    status=await session.status(),
                )
            elif message["type"] == "soulx.detach":
                session.detach()
                # The graph retains its output reader while the network client is away.
                session.attached.set()
            else:
                await session.handle(message)

        self.call(handle())
        payload.data = {"accepted": True}
        return payload

    def close(self, session_identity):
        session = self.sessions.pop(session_identity, None)
        if session:
            self.call(session.close())
        self.emitters.pop(session_identity, None)


def create_frontend_scheduler():
    return SessionScheduler(FrontendHooks(), max_open_sessions=16, max_concurrency=16)
