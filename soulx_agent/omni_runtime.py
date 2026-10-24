"""SGLang-Omni graph lifecycle and the benchmark/browser protocol adapter."""

import asyncio
import base64
import contextlib
import uuid
from pathlib import Path

import yaml
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import SessionLimits, TimedChunk
from starlette.websockets import WebSocketDisconnect


class OmniRuntime:
    def __init__(self, config_path):
        config = PipelineConfig.model_validate(yaml.safe_load(Path(config_path).read_text()))
        self.runner = MultiProcessPipelineRunner(config)

    async def start(self):
        await self.runner.start()

    async def close(self):
        await self.runner.stop()


class OmniVoiceSession:
    def __init__(self, runtime, session_id, config, send, web=False):
        self.runtime = runtime
        self.id = session_id
        self.config = config
        self.send = send
        self.web = web
        self.input_rate = config.get("input_rate", 24000)
        self.attached = asyncio.Event()
        self.attached.set()
        self.inputs = asyncio.Queue(maxsize=500)
        self.closed = False
        self.failure = None
        self.pending = {}
        self.workers = []
        self.sequence = 0
        self.append_lock = asyncio.Lock()
        self.identity = None

    async def start(self):
        coordinator = self.runtime.runner.coordinator
        self.identity = await coordinator.open_session(
            OmniRequest(
                None, params={"session_id": self.id, "config": self.config, "web": self.web}
            ),
            limits=SessionLimits(
                operation_timeout_s=60,
                idle_timeout_s=21600,
                max_pending_chunks=64,
                max_output_chunks=1024,
                max_output_bytes=32 * 1024 * 1024,
            ),
        )
        self.outputs = coordinator.session_outputs(self.identity)
        self.workers = [
            asyncio.create_task(self.receive()),
            asyncio.create_task(self.feed_browser()),
        ]

    async def feed_browser(self):
        while True:
            pcm = await self.inputs.get()
            await self.handle(
                {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
            )

    async def receive(self):
        try:
            async for chunk in self.outputs:
                if chunk.kind == "input_done":
                    continue
                if chunk.format != "soulx_event":
                    continue
                event = chunk.payload
                if event["type"] == "soulx.status":
                    future = self.pending.get(event["request_id"])
                    if future and not future.done():
                        future.set_result(event["status"])
                else:
                    if event["type"] == "error":
                        self.failure = event["error"].get("code", "runtime_failed")
                    while not self.closed:
                        await self.attached.wait()
                        try:
                            await self.send(event)
                            break
                        except (WebSocketDisconnect, OSError, RuntimeError):
                            self.attached.clear()
                            if event["type"] in {
                                "response.output_audio.delta",
                                "response.audio.delta",
                            }:
                                break
            if not self.closed:
                raise RuntimeError("Omni graph output ended before session close")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.failure = type(error).__name__
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(error)
            if self.attached.is_set():
                with contextlib.suppress(Exception):
                    await self.emit(
                        "error",
                        error={
                            "type": "server_error",
                            "code": "omni_failed",
                            "message": str(error),
                        },
                    )

    async def handle(self, message):
        if message.get("type") == "session.close":
            await self.close()
            return
        if self.failure:
            raise RuntimeError("Omni session failed: " + self.failure)
        async with self.append_lock:
            chunk = TimedChunk("control", 0, 0, self.sequence, message, format="soulx_event")
            self.sequence += 1
            await self.runtime.runner.coordinator.append_session(self.identity, chunk)

    async def emit(self, kind, **data):
        await self.attached.wait()
        await self.send({"type": kind, "event_id": uuid.uuid4().hex, **data})

    async def status(self):
        if self.failure:
            return {"busy": False, "session_id": self.id, "error": self.failure}
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self.handle({"type": "soulx.status", "request_id": request_id})
            return await asyncio.wait_for(future, timeout=10)
        finally:
            self.pending.pop(request_id, None)

    async def interrupt(self):
        await self.handle({"type": "response.cancel"})

    def detach(self):
        if self.closed:
            return
        self.attached.clear()
        task = asyncio.create_task(self.handle({"type": "soulx.detach"}))
        self.workers.append(task)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        for task in self.workers:
            task.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        if self.identity:
            try:
                await self.runtime.runner.coordinator.close_session(self.identity)
            finally:
                await self.outputs.aclose()
