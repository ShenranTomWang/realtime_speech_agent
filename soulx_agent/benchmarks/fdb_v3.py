"""Wall-clock audio replay and real upstream mock tools; no gold text enters the agent."""

import ast
import asyncio
import base64
import hashlib
import importlib
import inspect
import json
import os
import re
import sys
import tempfile
import time
import typing
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import soxr
import websockets
from pydantic import create_model


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def tool_definitions(registry):
    tools = []
    for name, function in registry.FUNCTIONS.items():
        hints = typing.get_type_hints(function)
        fields = {}
        for key, parameter in inspect.signature(function).parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[key] = (hints.get(key, typing.Any), default)
        schema = create_model(name + "Arguments", **fields).model_json_schema()
        tools.append(
            {
                "type": "function",
                "name": name,
                "description": name.replace("_", " "),
                "parameters": schema,
            }
        )
    return tools


def upstream_policy(source):
    tree = ast.parse((source / "lk_agent_tool.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VoiceAgent":
            for child in ast.walk(node):
                if isinstance(child, ast.keyword) and child.arg == "instructions":
                    return ast.literal_eval(child.value)
    raise ValueError("Pinned FDB VoiceAgent instructions not found")


async def infer_one(
    source, audio_path, output, scenario_id, speaker_id, tail_seconds, latency_profile
):
    from mock_apis import MockAPIRegistry

    output.mkdir(parents=True, exist_ok=True)
    registry = MockAPIRegistry(latency_profile=latency_profile)
    url = os.environ.get("SOULX_SERVER_URL", "ws://127.0.0.1:8765/v1/realtime")
    samples, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    samples = soxr.resample(samples, rate, 24000)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    result = {
        "example_id": scenario_id,
        "pid": speaker_id,
        "provider": "soulx_claude_code",
        "transport": "direct_websocket_realtime",
        "latency_profile": latency_profile,
        "input_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "actual_tool_calls": [],
        "events": [],
        "status": "running",
    }
    recorded = bytearray()
    tool_tasks = set()
    session_id = None
    start = None
    async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": upstream_policy(source),
                        "tools": tool_definitions(registry),
                    },
                }
            )
        )
        while True:
            initial = json.loads(await ws.recv())
            if initial["type"] == "session.updated":
                session_id = initial["session"]["id"]
                break
            if initial["type"] == "error":
                raise RuntimeError(initial["error"])
        start = time.monotonic()
        result["stream_start_time"] = time.time()

        async def execute(event):
            args = json.loads(event["arguments"])
            begin = time.monotonic() - start
            value = await asyncio.to_thread(registry.call, event["name"], **args)
            result["actual_tool_calls"].append(
                {
                    "function": event["name"],
                    "args": args,
                    "result": value,
                    "timestamp_start": begin,
                    "timestamp_end": time.monotonic() - start,
                }
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": event["call_id"],
                            "output": json.dumps(value),
                        },
                    }
                )
            )

        async def receive():
            async for raw in ws:
                event = json.loads(raw)
                kind = event["type"]
                now = time.monotonic() - start
                if kind == "response.output_audio.delta":
                    audio = base64.b64decode(event["delta"])
                    offset = round(now * 24000) * 2
                    if len(recorded) < offset:
                        recorded.extend(bytes(offset - len(recorded)))
                    recorded.extend(audio)
                elif kind == "response.function_call_arguments.done":
                    task = asyncio.create_task(execute(event))
                    tool_tasks.add(task)
                elif kind == "error":
                    raise RuntimeError(event["error"])
                else:
                    result["events"].append({"time": now, **event})

        receiver = asyncio.create_task(receive())
        try:
            # Continuous audio, including trailing silence, lets SoulX decide every turn.
            payload = pcm + bytes(round(tail_seconds * 24000) * 2)
            for offset in range(0, len(payload), 4800):
                if receiver.done():
                    await receiver
                    raise RuntimeError("Realtime stream ended early")
                for task in tool_tasks:
                    if task.done():
                        task.result()
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(payload[offset : offset + 4800]).decode(),
                        }
                    )
                )
                await asyncio.sleep(max(0, start + (offset + 4800) / 48000 - time.monotonic()))
            if any(not task.done() for task in tool_tasks):
                raise TimeoutError("Tool work exceeded recording window")
            # Reject unfinished backend/foreground work instead of silently scoring truncation.
            base = (
                url.split("/v1/realtime")[0]
                .replace("ws://", "http://")
                .replace("wss://", "https://")
            )
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.get(base + "/sessions/" + session_id)
                response.raise_for_status()
                if response.json().get("error"):
                    raise RuntimeError("Voice runtime failed")
                if response.json()["busy"]:
                    raise TimeoutError("Agent work exceeded recording window")
            result["status"] = "inference_completed"
        except Exception as error:
            result["status"] = "inference_failed"
            result["error"] = type(error).__name__
            result["error_message"] = str(error)
            raise
        finally:
            receiver.cancel()
            for task in tool_tasks:
                task.cancel()
            await asyncio.gather(receiver, *tool_tasks, return_exceptions=True)
            sf.write(
                output / "output_soulx_claude_code.wav", np.frombuffer(recorded, dtype="<i2"), 24000
            )
            atomic_json(output / "result_soulx_claude_code.json", result)
            base = (
                url.split("/v1/realtime")[0]
                .replace("ws://", "http://")
                .replace("wss://", "https://")
            )
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.delete(base + "/sessions/" + session_id)
                response.raise_for_status()


async def infer(source, data_dir, output_dir, tail_seconds=120, latency_profile="instant"):
    sys.path.insert(0, str(source))
    scenarios = json.loads((source / "benchmark_data_v2.json").read_text())["scenarios"]
    ids = {item["id"] for item in scenarios}
    files = sorted(data_dir.glob("*/input.wav"))
    if not files:
        raise FileNotFoundError("FDB v3 data missing: expected <scenario>_<speaker>/input.wav")
    for audio in files:
        match = re.fullmatch(r"(.+)_([0-9a-f]{24})", audio.parent.name)
        if not match or match[1] not in ids:
            raise ValueError(f"Unexpected FDB data directory: {audio.parent.name}")
        output = output_dir / audio.parent.name
        result_file = output / "result_soulx_claude_code.json"
        if result_file.exists():
            previous = json.loads(result_file.read_text())
            if previous["input_sha256"] != hashlib.sha256(audio.read_bytes()).hexdigest():
                raise ValueError("Refusing to resume changed audio")
            if previous.get("latency_profile") != latency_profile:
                raise ValueError("Refusing to resume changed latency profile")
            if previous["status"] in {"inference_completed", "completed"}:
                continue
        await infer_one(source, audio, output, match[1], match[2], tail_seconds, latency_profile)


def prepare_scoring(source, data_dir, output_dir):
    sys.path.insert(0, str(source))
    upstream = importlib.import_module("run_tool_benchmark")
    model = upstream.load_asr_model()
    files = sorted(output_dir.glob("*/result_soulx_claude_code.json"))
    if not files:
        raise FileNotFoundError("No FDB inference results")
    for path in files:
        result = json.loads(path.read_text())
        if result["status"] == "completed":
            continue
        if result["status"] != "inference_completed":
            raise RuntimeError(f"Failed inference must be rerun before scoring: {path}")
        input_path = data_dir / path.parent.name / "input.wav"
        with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
            if sf.info(input_path).channels > 1:
                samples, rate = sf.read(input_path, dtype="float32", always_2d=True)
                mono_path = Path(temporary) / "input.wav"
                sf.write(mono_path, samples.mean(axis=1), rate)
                input_path = mono_path
            input_asr = upstream.run_asr(model, input_path)
        output_asr = upstream.run_asr(model, path.parent / "output_soulx_claude_code.wav")
        if not input_asr["chunks"]:
            raise RuntimeError("Input ASR failed; cannot estimate benchmark speech-end timing")
        # Match upstream's first long-gap rule for the user's initial spoken segment.
        end = input_asr["chunks"][-1]["timestamp"][1]
        for current, following in zip(input_asr["chunks"], input_asr["chunks"][1:]):
            if following["timestamp"][0] - current["timestamp"][1] > 2:
                end = current["timestamp"][1]
                break
        result.update(
            user_speech_end_rel=end,
            transcript=output_asr["text"],
            asr_chunks=output_asr["chunks"],
            status="completed",
        )
        if not output_asr["chunks"] and any(
            event["type"] == "response.output_audio_transcript.done" for event in result["events"]
        ):
            raise RuntimeError(
                "Output ASR failed on generated speech; do not substitute model text"
            )
        atomic_json(path, result)
