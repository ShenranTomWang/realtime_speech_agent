"""Realtime protocol bridge shared by the SoulX UI and benchmark-owned tool clients."""

import asyncio
import base64
import contextlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .application import create_harness
from .audio import SoulXAudio, synthesize
from .contracts import Tool
from .orchestration import Coordinator

SESSIONS = {}
OMNI_RUNTIME = None


class VoiceSession:
    def __init__(self, session_id, config, send, web=False):
        self.id = session_id
        self.send = send
        self.web = web
        self.config = config
        self.user_speaking = False
        self.epoch = 0
        self.input_samples = 0
        self.input_rate = config.get("input_rate", 24000)
        self.audio = SoulXAudio(self.input_rate)
        self.inputs = asyncio.Queue(maxsize=500)
        self.replies = asyncio.Queue()
        self.pending_tools = {}
        self.tool_lock = asyncio.Lock()
        self.workers = set()
        self.playback = None
        self.pending_text = None
        self.closed = False
        self.failure = None
        self.attached = asyncio.Event()
        self.attached.set()
        definitions = config.get("tools", [])
        names = [item["name"] for item in definitions]
        if len(names) != len(set(names)) or set(names) & {
            "submit_task",
            "task_status",
            "cancel_task",
        }:
            raise ValueError("Duplicate or reserved environment tool name")
        tools = []
        for definition in definitions:

            async def invoke(args, definition=definition):
                return await self.call_tool(definition["name"], args)

            tools.append(
                Tool(
                    definition["name"],
                    definition.get("description", ""),
                    definition.get("parameters", {"type": "object", "properties": {}}),
                    invoke,
                )
            )
        root = Path(os.environ.get("SOULX_RUN_DIR", "runs"))
        self.coordinator = Coordinator(
            config.get("instructions", ""),
            tools,
            root / self.id / "events.jsonl",
            harness=create_harness,
            frontend_tools_override=os.environ.get("SOULX_FRONTEND_TOOLS_OVERRIDE"),
        )

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.workers.add(task)
        task.add_done_callback(self.workers.discard)
        task.add_done_callback(self.check_worker)
        return task

    def check_worker(self, task):
        if not task.cancelled() and task.exception():
            self.failure = type(task.exception()).__name__
            self.coordinator.record("runtime.failed", error=self.failure)
            if not self.closed and self.attached.is_set():

                async def report_failure():
                    with contextlib.suppress(Exception):
                        await self.emit(
                            "error",
                            error={
                                "type": "server_error",
                                "code": "runtime_failed",
                                "message": "Agent runtime failed; inspect its journal.",
                            },
                        )

                asyncio.create_task(report_failure())

    async def emit(self, kind, **data):
        await self.attached.wait()
        try:
            await self.send({"type": kind, "event_id": uuid.uuid4().hex, **data})
        except (WebSocketDisconnect, OSError, RuntimeError) as error:
            self.attached.clear()
            raise asyncio.CancelledError from error

    async def start(self):
        await self.audio.connect()
        self.spawn(self.consume_audio())
        self.spawn(self.deliver())
        self.spawn(self.backend_results())

    async def call_tool(self, name, args):
        async with self.tool_lock:
            call_id = uuid.uuid4().hex
            future = asyncio.get_running_loop().create_future()
            self.pending_tools[call_id] = future
            self.coordinator.record("tool.started", call_id=call_id, name=name, arguments=args)
            try:
                await self.emit(
                    "response.function_call_arguments.done",
                    call_id=call_id,
                    name=name,
                    arguments=json.dumps(args),
                    item_id=call_id,
                    output_index=0,
                )
                result = await asyncio.wait_for(future, timeout=180)
                self.coordinator.record("tool.completed", call_id=call_id, name=name, result=result)
                return result
            finally:
                self.pending_tools.pop(call_id, None)

    async def consume_audio(self):
        while True:
            pcm = await self.inputs.get()
            self.input_samples += len(pcm) // 2
            state = await self.audio.process(pcm)
            milliseconds = int(self.input_samples * 1000 / self.input_rate)
            if state["state"] == "nonidle" and not self.user_speaking:
                self.user_speaking = True
                self.epoch += 1
                self.user_item = uuid.uuid4().hex
                await self.interrupt()
                await self.emit(
                    "input_audio_buffer.speech_started",
                    audio_start_ms=milliseconds,
                    item_id=self.user_item,
                )
            elif state["state"] == "speak":
                self.user_speaking = False
                item_id = getattr(self, "user_item", uuid.uuid4().hex)
                await self.emit(
                    "input_audio_buffer.speech_stopped", audio_end_ms=milliseconds, item_id=item_id
                )
                text = state.get("text", "").strip()
                if text:
                    await self.emit(
                        "conversation.item.input_audio_transcription.completed",
                        transcript=text,
                        item_id=item_id,
                        content_index=0,
                    )
                    self.spawn(self.respond(text, self.epoch))

    async def respond(self, text, epoch):
        answer = await self.coordinator.turn(text)
        if answer and epoch == self.epoch:
            await self.replies.put((None, answer, epoch))

    async def backend_results(self):
        while True:
            task_id, text = await self.coordinator.delivery.get()
            await self.replies.put((task_id, text, None))

    async def deliver(self):
        while True:
            task_id, text, epoch = await self.replies.get()
            while (
                self.user_speaking
                or self.coordinator.front_lock.locked()
                or not self.attached.is_set()
            ):
                await asyncio.sleep(0.05)
            if task_id is None and epoch != self.epoch:
                continue
            self.playback = asyncio.create_task(self.speak(text, epoch))
            try:
                await self.playback
                self.coordinator.record("speech.emitted", task_id=task_id, text=text)
                if task_id:
                    self.coordinator.history.append({"role": "assistant", "content": text})
            except asyncio.CancelledError:
                if self.closed:
                    raise
                # Accepted backend results remain pending after a barge-in.
                if task_id:
                    await self.replies.put((task_id, text, None))
            finally:
                self.playback = None

    async def speak(self, text, epoch=None):
        pcm = await synthesize(text)
        if self.user_speaking or (epoch is not None and epoch != self.epoch):
            raise asyncio.CancelledError
        item_id = uuid.uuid4().hex
        response_id = uuid.uuid4().hex
        item = {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        await self.emit(
            "response.created", response={"id": response_id, "status": "in_progress", "output": []}
        )
        await self.emit(
            "response.output_item.added",
            response_id=response_id,
            output_index=0,
            item=item,
            item_id=item_id,
        )
        await self.emit(
            "response.output_audio_transcript.delta",
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            content_index=0,
            delta=text,
        )
        for offset in range(0, len(pcm), 4800):
            await self.emit(
                "response.output_audio.delta",
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=0,
                delta=base64.b64encode(pcm[offset : offset + 4800]).decode(),
            )
            await asyncio.sleep(len(pcm[offset : offset + 4800]) / 48000)
        await self.emit(
            "response.output_audio.done",
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            content_index=0,
        )
        await self.emit(
            "response.output_audio_transcript.done",
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            content_index=0,
            transcript=text,
        )
        item.update(status="completed", content=[{"type": "audio", "transcript": text}])
        await self.emit(
            "response.output_item.done",
            response_id=response_id,
            output_index=0,
            item=item,
            item_id=item_id,
        )
        await self.emit(
            "response.done", response={"id": response_id, "status": "completed", "output": [item]}
        )

    async def interrupt(self):
        if self.playback and not self.playback.done():
            self.playback.cancel()
            await self.emit("response.cancelled")
        self.coordinator.record("speech.interrupted")

    async def handle(self, message):
        kind = message.get("type")
        if kind == "input_audio_buffer.append":
            pcm = base64.b64decode(message["audio"], validate=True)
            if len(pcm) % 2:
                raise ValueError("Expected PCM16 mono audio")
            self.inputs.put_nowait(pcm)
        elif kind == "conversation.item.create":
            item = message["item"]
            if item["type"] == "function_call_output":
                future = self.pending_tools.get(item["call_id"])
                if future and not future.done():
                    future.set_result(item["output"])
            elif item["type"] == "message" and item.get("role") == "user":
                self.pending_text = " ".join(block.get("text", "") for block in item["content"])
        elif kind == "response.create" and self.pending_text:
            text, self.pending_text = self.pending_text, None
            self.spawn(self.respond(text, self.epoch))
        elif kind in {"response.cancel", "conversation.item.truncate"}:
            await self.interrupt()
        elif kind == "session.close":
            await self.close()

    async def status(self):
        busy = (
            self.coordinator.front_lock.locked()
            or any(item["status"] == "running" for item in self.coordinator.tasks.values())
            or bool(self.playback)
            or not self.replies.empty()
            or not self.coordinator.delivery.empty()
        )
        return {"busy": busy, "session_id": self.id, "error": self.failure}

    def detach(self):
        self.attached.clear()
        self.user_speaking = False
        if self.playback and not self.playback.done():
            self.playback.cancel()
        self.coordinator.record("client.detached")

    async def close(self):
        self.closed = True
        if self.playback:
            self.playback.cancel()
        for task in list(self.workers):
            task.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        await self.coordinator.close()
        await self.audio.close()


def make_session(session_id, config, send, web=False):
    if OMNI_RUNTIME is not None:
        from .omni_runtime import OmniVoiceSession

        return OmniVoiceSession(OMNI_RUNTIME, session_id, config, send, web=web)
    return VoiceSession(session_id, config, send, web=web)


@asynccontextmanager
async def lifespan(app):
    global OMNI_RUNTIME
    if os.environ.get("SOULX_FRONTEND_RUNTIME", "soulx") == "sglang_omni":
        from .omni_runtime import OmniRuntime

        OMNI_RUNTIME = OmniRuntime(os.environ["SOULX_OMNI_CONFIG"])
        await OMNI_RUNTIME.start()
    try:
        yield
    finally:
        await asyncio.gather(
            *(session.close() for session in SESSIONS.values()), return_exceptions=True
        )
        if OMNI_RUNTIME is not None:
            await OMNI_RUNTIME.close()
            OMNI_RUNTIME = None


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def health():
    return {
        "status": "ok",
        "sessions": len(SESSIONS),
        "runtime": "sglang_omni" if OMNI_RUNTIME else "soulx",
    }


@app.get("/sessions/{session_id}")
async def session_status(session_id: str):
    session = SESSIONS[session_id]
    return await session.status()


@app.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    session = SESSIONS.pop(session_id, None)
    if session:
        await session.close()
    return {"closed": bool(session)}


@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket):
    await ws.accept()
    session = None
    try:
        await ws.send_json(
            {
                "type": "session.created",
                "session": {"id": uuid.uuid4().hex, "object": "realtime.session"},
            }
        )
        while True:
            message = await ws.receive_json()
            if message.get("type") == "session.update":
                if session:
                    raise ValueError("Session configuration cannot change after initialization")
                config = message.get("session", message)
                input_format = config.get("audio", {}).get("input", {}).get("format", {})
                if input_format and (
                    input_format.get("type") != "audio/pcm"
                    or input_format.get("rate", 24000) != 24000
                ):
                    raise ValueError("This endpoint requires PCM16 mono at 24000 Hz")
                session_id = ws.query_params.get("session_id") or uuid.uuid4().hex
                uuid.UUID(hex=session_id)
                if session_id in SESSIONS:
                    session = SESSIONS[session_id]
                    if session.web:
                        raise ValueError("Cannot change a browser session's transport")
                    if session.attached.is_set():
                        raise ValueError("Session already has an attached client")
                    session.send = ws.send_json
                    session.attached.set()
                else:
                    session = make_session(session_id, config, ws.send_json)
                    await session.start()
                    SESSIONS[session_id] = session
                await session.emit(
                    "session.updated",
                    session={"id": session.id, "object": "realtime.session", **config},
                )
            elif session:
                await session.handle(message)
                if session.closed:
                    SESSIONS.pop(session.id, None)
                    break
    except WebSocketDisconnect:
        pass
    except Exception as error:
        with contextlib.suppress(Exception):
            await ws.send_json(
                {"type": "error", "error": {"message": str(error), "type": "runtime_error"}}
            )
    finally:
        if session:
            session.detach()
            # Backend work survives a transport disconnect. Reattach with session_id,
            # or explicitly DELETE the session when an evaluation episode ends.


@app.websocket("/ws")
async def browser(ws: WebSocket):
    await ws.accept()
    session = None

    async def send(message):
        kind = message["type"]
        if kind == "response.output_audio.delta":
            await ws.send_bytes(base64.b64decode(message["delta"]))
        elif kind == "response.cancelled":
            await ws.send_json({"event": "stop_audio", "data": {}})
        elif kind == "conversation.item.input_audio_transcription.completed":
            await ws.send_json(
                {"event": "user_transcription", "data": {"text": message["transcript"]}}
            )
        elif kind == "response.output_audio_transcript.done":
            await ws.send_json({"event": "text_response", "data": {"text": message["transcript"]}})
        elif kind == "error":
            await ws.send_json(
                {
                    "event": "vad_status",
                    "data": {"state": "error", "message": message["error"]["message"]},
                }
            )
        elif kind in {"response.created", "response.done", "input_audio_buffer.speech_started"}:
            status = {
                "response.created": "SPEAKING",
                "response.done": "READY",
                "input_audio_buffer.speech_started": "LISTENING",
            }[kind]
            await ws.send_json({"event": "circle_status", "data": {"status": status}})

    try:
        await ws.send_json({"event": "connect_ack", "data": {}})
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") and session:
                session.inputs.put_nowait(message["bytes"])
            elif message.get("text"):
                config = json.loads(message["text"])
                if config["event"] == "config_audio":
                    sample_rate = config["data"]["sample_rate"]
                    if session is None:
                        session_id = ws.query_params.get("session_id") or uuid.uuid4().hex
                        uuid.UUID(hex=session_id)
                        if session_id in SESSIONS:
                            session = SESSIONS[session_id]
                            if not session.web or session.attached.is_set():
                                raise ValueError("Session already has an attached client")
                            if session.input_rate != sample_rate:
                                raise ValueError(
                                    "Resumed session requires the same microphone sample rate"
                                )
                            session.send = send
                            session.attached.set()
                        else:
                            session = make_session(
                                session_id, {"input_rate": sample_rate}, send, web=True
                            )
                            await session.start()
                            SESSIONS[session_id] = session
                    await ws.send_json({"event": "vad_loading", "data": {"loading": False}})
                elif config["event"] == "duplex_stop" and session:
                    session.user_speaking = False
                    await session.interrupt()
    finally:
        if session:
            session.detach()


@app.middleware("http")
async def browser_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response


WEB = Path(__file__).resolve().parents[1] / "web"
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
