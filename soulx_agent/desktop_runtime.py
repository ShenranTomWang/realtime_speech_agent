"""Laptop-local audio, task orchestration and Claude Code permission requests."""

import asyncio
import contextlib
import hmac
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from . import server
from .audio import SoulXAudio, synthesize
from .contracts import ModelConnection
from .harness import ClaudeHarness
from .toolchain import load_tools

SESSION = None
PERMISSIONS = {}
SEND = None


async def permission(tool_name, input_data, context):
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    request_id = uuid.uuid4().hex
    future = asyncio.get_running_loop().create_future()
    PERMISSIONS[request_id] = {
        "future": future, "tool": tool_name, "input": input_data,
    }
    try:
        if SEND:
            with contextlib.suppress(WebSocketDisconnect, OSError, RuntimeError):
                await SEND(
                    {"type": "permission.request", "id": request_id, "tool": tool_name, "input": input_data}
                )
        allowed = await future
        if allowed:
            return PermissionResultAllow(updated_input=input_data)
        return PermissionResultDeny(message="The user denied this action.")
    finally:
        PERMISSIONS.pop(request_id, None)
        if SEND:
            with contextlib.suppress(WebSocketDisconnect, OSError, RuntimeError):
                await SEND({"type": "permission.resolved", "id": request_id})


def create_harness(role, prompt, tools):
    return ClaudeHarness(
        ModelConnection(role, os.environ["SOULX_GATEWAY_URL"], "SOULX_GATEWAY_KEY"),
        prompt, tools, native_tools=role == "backend",
        cwd=os.environ["SOULX_WORKING_DIR"],
        can_use_tool=permission if role == "backend" else None,
    )


@asynccontextmanager
async def lifespan(app):
    server.create_harness = create_harness
    server.SoulXAudio = lambda input_rate: SoulXAudio(input_rate, os.environ["SOULX_MODEL_TOKEN"])
    server.synthesize = lambda text: synthesize(text, os.environ["SOULX_MODEL_TOKEN"])
    try:
        yield
    finally:
        if SESSION:
            await SESSION.close()
        for item in PERMISSIONS.values():
            if not item["future"].done():
                item["future"].set_result(False)


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def desktop(ws: WebSocket):
    global SESSION, SEND
    expected_origin = os.environ["SOULX_DESKTOP_ORIGIN"]
    if ws.headers.get("origin") != expected_origin:
        await ws.close(code=1008)
        return
    await ws.accept()
    snapshots = None
    attached = False
    try:
        auth = await asyncio.wait_for(ws.receive_json(), timeout=10)
        if not hmac.compare_digest(str(auth.get("token", "")), os.environ["SOULX_DESKTOP_KEY"]):
            await ws.close(code=1008)
            return
        if SEND is not None:
            await ws.close(code=1008, reason="Another desktop is already attached")
            return
        SEND = ws.send_json
        attached = True
        if SESSION is not None:
            SESSION.send = ws.send_json
            SESSION.attached.set()
        await ws.send_json({"type": "desktop.ready"})
        for request_id, item in PERMISSIONS.items():
            await ws.send_json(
                {"type": "permission.request", "id": request_id, "tool": item["tool"], "input": item["input"]}
            )

        async def publish_tasks():
            previous = None
            while True:
                if SESSION:
                    tasks = [
                        {"id": key, **{
                            name: value for name, value in item.items() if name not in {"agent", "future"}
                        }} for key, item in SESSION.coordinator.tasks.items()
                    ]
                    if tasks != previous:
                        await ws.send_json({"type": "tasks.updated", "tasks": tasks})
                        previous = tasks
                await asyncio.sleep(0.25)

        snapshots = asyncio.create_task(publish_tasks())
        while True:
            message = await ws.receive_json()
            kind = message.get("type")
            if kind == "session.start":
                if SESSION is None:
                    SESSION = server.VoiceSession(
                        uuid.uuid4().hex,
                        {"instructions": "Operate on the laptop using Claude Code permissions. Ask when unclear."},
                        ws.send_json,
                    )
                    try:
                        await SESSION.start()
                    except Exception:
                        await SESSION.close()
                        SESSION = None
                        raise
                await ws.send_json({"type": "session.ready"})
            elif kind == "permission.answer":
                item = PERMISSIONS.get(message.get("id"))
                if item and not item["future"].done():
                    item["future"].set_result(message.get("allow") is True)
            elif kind == "task.cancel" and SESSION:
                task = SESSION.coordinator.tasks.get(message["id"])
                if task and task["status"] == "running":
                    task["status"] = "cancelled"
                    task["future"].cancel()
                    SESSION.coordinator.record("task.cancelled", task_id=message["id"])
            elif kind == "task.steer" and SESSION:
                tool = next(tool for tool in load_tools(SESSION.coordinator) if tool.name == "steer_task")
                result = await tool.call({"task_id": message["id"], "instruction": message["instruction"]})
                await ws.send_json({"type": "task.steered", "result": result})
            elif SESSION:
                await SESSION.handle(message)
    except WebSocketDisconnect:
        pass
    except Exception as error:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "error": {"message": str(error)}})
    finally:
        if snapshots:
            snapshots.cancel()
            await asyncio.gather(snapshots, return_exceptions=True)
        if attached:
            SEND = None
            if SESSION:
                SESSION.detach()


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[1] / "desktop", html=True))
