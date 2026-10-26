"""Own all service processes in one allocation; fail if any service exits."""

import argparse
import asyncio
import json
import os
import secrets
import signal
import subprocess
import time
from pathlib import Path

import httpx
import websockets

from soulx_agent.llm_runner import DEFAULT_CONFIG, load_llm_plan
from soulx_agent.processes import launch, service_devices, stop_processes
from soulx_agent.provenance import verify_run

ROOT = Path(__file__).resolve().parents[1]


async def wait_voice_runtime(processes, port):
    timeout = float(os.environ.get("SOULX_STARTUP_TIMEOUT", "1800"))
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("Voice runtime exited during graph initialization")
            try:
                response = await client.get(f"http://127.0.0.1:{port}/healthz")
                response.raise_for_status()
                print("Voice runtime graph ready", flush=True)
                return
            except httpx.HTTPError:
                await asyncio.sleep(2)
    raise TimeoutError("Voice runtime graph initialization timed out")


async def wait_services(processes, gateway_url, llm_ready):
    timeout = float(os.environ.get("SOULX_STARTUP_TIMEOUT", "1800"))
    deadline = time.monotonic() + timeout
    last_report = 0
    pending = "LLM readiness marker"
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("A service exited during startup")
            try:
                pending = "LLM readiness marker"
                if not llm_ready.is_file():
                    raise OSError("LLM runner is not ready")
                for url in [
                    gateway_url + "/health/liveliness",
                    os.environ["SOULX_TTS_URL"].removesuffix("/tts") + "/health",
                    os.environ["SOULX_SERVER_URL"]
                    .replace("ws://", "http://")
                    .removesuffix("/v1/realtime")
                    + "/healthz",
                ]:
                    pending = url
                    response = await client.get(url)
                    response.raise_for_status()
                pending = os.environ["SOULX_VAD_URL"]
                async with websockets.connect(os.environ["SOULX_VAD_URL"]):
                    pass
                return
            except (httpx.HTTPError, OSError, websockets.WebSocketException) as error:
                if time.monotonic() - last_report >= 30:
                    print(f"Startup waiting for {pending}: {type(error).__name__}", flush=True)
                    last_report = time.monotonic()
                await asyncio.sleep(2)
    raise TimeoutError(f"Service startup exceeded {timeout:g}s; waiting for {pending}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", choices=["tau_voice", "eva", "fdb_v3", "serve"], required=True
    )
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument(
        "--llm_config", type=Path, default=os.environ.get("SOULX_LLM_CONFIG", DEFAULT_CONFIG)
    )
    parser.add_argument(
        "--frontend_runtime", choices=["soulx", "sglang_omni"], default="sglang_omni"
    )
    parser.add_argument("--omni_config", type=Path, default=ROOT / "config/omni.yaml")
    parser.add_argument("--gpu_layout", choices=["split", "shared"], default="split")
    parser.add_argument("--caller_mode", choices=["local", "service"], default="local")
    parser.add_argument("--frontend_tools_override", type=Path, default=None)
    args = parser.parse_args()
    llm_device, speech_device = service_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES", ""), args.gpu_layout
    )
    print(f"GPU layout={args.gpu_layout}: LLM={llm_device}, speech={speech_device}", flush=True)
    if args.frontend_tools_override is not None:
        from soulx_agent.frontend_tools import load_override

        load_override(args.frontend_tools_override)
        os.environ["SOULX_FRONTEND_TOOLS_OVERRIDE"] = str(args.frontend_tools_override.resolve())
    else:
        os.environ.pop("SOULX_FRONTEND_TOOLS_OVERRIDE", None)
    os.environ["SOULX_CALLER_MODE"] = args.caller_mode
    port_offset = int(
        os.environ.setdefault(
            "SOULX_PORT_OFFSET",
            str({"tau_voice": 0, "eva": 100, "fdb_v3": 200, "serve": 300}[args.benchmark]),
        )
    )
    vad_port, tts_port, voice_port = (8000 + port_offset, 6006 + port_offset, 8765 + port_offset)
    os.environ["SOULX_VAD_URL"] = f"ws://127.0.0.1:{vad_port}/turn"
    os.environ["SOULX_TTS_URL"] = f"http://127.0.0.1:{tts_port}/tts"
    os.environ["SOULX_SERVER_URL"] = f"ws://127.0.0.1:{voice_port}/v1/realtime"
    os.environ["SOULX_FRONTEND_RUNTIME"] = args.frontend_runtime
    os.environ["SOULX_OMNI_CONFIG"] = str(args.omni_config.resolve())
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.llm_config = args.llm_config.resolve()
    plan = load_llm_plan(args.llm_config)
    if not os.environ.get("SOULX_GATEWAY_KEY"):
        os.environ["SOULX_GATEWAY_KEY"] = "sk-" + secrets.token_urlsafe(32)
    plan.validate_credentials()
    os.environ["SOULX_GATEWAY_URL"] = plan.gateway_url
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    for name, item in lock.items():
        revision = subprocess.check_output(
            ["git", "-C", str(args.source_dir / name), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != item["revision"]:
            raise RuntimeError(f"Source revision mismatch: {name}")
    verify_run(args.output_dir, args.source_dir, args.benchmark, args.llm_config)
    os.environ["SOULX_RUN_DIR"] = str(args.output_dir / "sessions")
    (args.output_dir / "sources.lock.json").write_text(json.dumps(lock, indent=2))
    soulx = args.source_dir / "SoulX-Duplug"
    llm_dir = args.output_dir / "llm"
    llm_ready = llm_dir / "ready.json"
    llm_ready.unlink(missing_ok=True)
    specs = [
        (
            "soulx-agent",
            [
                "env",
                "CUDA_VISIBLE_DEVICES=" + llm_device,
                "python",
                "-m",
                "scripts.run_llm",
                "--llm_config",
                str(args.llm_config),
                "--output_dir",
                str(llm_dir),
            ],
            ROOT,
        ),
        (
            "soulx-vad",
            [
                "env",
                "CUDA_VISIBLE_DEVICES=" + speech_device,
                "uvicorn",
                "server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(vad_port),
            ],
            soulx,
        ),
        (
            "soulx-tts",
            [
                "env",
                "CUDA_VISIBLE_DEVICES=" + speech_device,
                "PYTHONPATH=" + os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]),
                "python",
                "-m",
                "scripts.serve_tts",
                "--host",
                "127.0.0.1",
                "--port",
                str(tts_port),
                "--model_dir",
                str(soulx / "pretrained_models/Index-TTS-1.5-vLLM"),
            ],
            soulx / "dialogue_system/modules/index_tts_vllm",
        ),
        (
            "soulx-omni" if args.frontend_runtime == "sglang_omni" else "soulx-agent",
            [
                "env",
                "SOULX_GATEWAY_URL=" + plan.gateway_url,
                "SOULX_RUN_DIR=" + str(args.output_dir / "sessions"),
                "python",
                "-m",
                "scripts.serve",
                "--port",
                str(voice_port),
                "--frontend_runtime",
                args.frontend_runtime,
                "--omni_config",
                str(args.omni_config.resolve()),
                *(
                    ["--frontend_tools_override", str(args.frontend_tools_override.resolve())]
                    if args.frontend_tools_override is not None
                    else []
                ),
            ],
            ROOT,
        ),
    ]
    processes = []

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.benchmark in {"tau_voice", "eva"}:
            if args.data_dir is None:
                raise ValueError("Voice benchmark inference requires --data_dir")
            preflight = launch(
                "soulx-tau" if args.benchmark == "tau_voice" else "soulx-eva",
                [
                    "python",
                    "-m",
                    "scripts.check_benchmark_setup",
                    "--benchmark",
                    args.benchmark,
                    "--dataset_dir",
                    str(args.data_dir.resolve().parent),
                ],
                ROOT,
            )
            processes.append(preflight)
            if preflight.wait() != 0:
                raise RuntimeError("Benchmark data preflight failed before model startup")
            processes.remove(preflight)
        # Initialize the CPU graph before competing model loaders occupy the CPUs.
        environment, command, cwd = specs[-1]
        processes.append(launch(environment, command, cwd))
        asyncio.run(wait_voice_runtime(processes, voice_port))
        for environment, command, cwd in specs[:-1]:
            processes.append(launch(environment, command, cwd))
        asyncio.run(wait_services(processes, plan.gateway_url, llm_ready))
        smoke = launch(
            "soulx-agent",
            [
                "python",
                "-m",
                "scripts.smoke_agents",
                "--llm_config",
                str(args.llm_config),
                "--output_dir",
                str(args.output_dir),
            ],
            ROOT,
        )
        processes.append(smoke)
        if smoke.wait() != 0:
            raise RuntimeError("Real frontend/backend harness smoke check failed")
        processes.remove(smoke)
        voice_smoke = launch(
            "soulx-agent",
            [
                "python",
                "-m",
                "scripts.smoke_voice",
                "--server_url",
                os.environ["SOULX_SERVER_URL"],
                "--output_dir",
                str(args.output_dir),
            ],
            ROOT,
        )
        processes.append(voice_smoke)
        if voice_smoke.wait() != 0:
            raise RuntimeError("Real voice session smoke check failed")
        processes.remove(voice_smoke)
        worker = None
        if args.benchmark != "serve":
            command = [
                "python",
                "-m",
                "scripts.run_benchmark",
                args.benchmark,
                "inference",
                "--caller_mode",
                args.caller_mode,
                "--llm_config",
                str(args.llm_config),
                "--source_dir",
                str(args.source_dir),
                "--output_dir",
                str(args.output_dir),
            ]
            if args.data_dir:
                command.extend(["--data_dir", str(args.data_dir)])
            worker = launch(
                "soulx-" + {"tau_voice": "tau", "eva": "eva", "fdb_v3": "fdb"}[args.benchmark],
                command,
                ROOT,
            )
            processes.append(worker)
        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    if process is worker and code == 0:
                        return
                    raise RuntimeError(f"Stack child exited with status {code}")
            time.sleep(1)
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    main()
