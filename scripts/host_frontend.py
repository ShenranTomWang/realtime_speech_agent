"""Own GPU model services and a temporary authenticated Cloudflare ingress."""

import argparse
import os
import re
import secrets
import shlex
import signal
import subprocess
import tarfile
import time
from pathlib import Path

from soulx_agent.frontend_tools import load_override
from soulx_agent.llm_runner import load_llm_plan
from soulx_agent.processes import launch, service_devices, stop_processes

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--llm_config", type=Path, default=ROOT / "config/llm/qwen35_4b_sglang_vector.yaml"
    )
    parser.add_argument(
        "--frontend_tools_override", type=Path,
        default=ROOT / "config/frontend_tools/delegate_simple.yaml",
    )
    parser.add_argument("--port", type=int, default=19065)
    parser.add_argument("--gpu_layout", choices=["shared", "split"], default="shared")
    parser.add_argument("--cloudflared", default="cloudflared")
    args = parser.parse_args()
    load_override(args.frontend_tools_override)
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)
    launcher = args.output_dir / "launch_laptop.sh"
    launcher.unlink(missing_ok=True)
    port_offset = int(os.environ.setdefault("SOULX_PORT_OFFSET", "10000"))
    vad_port, tts_port = 8000 + port_offset, 6006 + port_offset
    plan = load_llm_plan(args.llm_config)
    runner = plan.runners[plan.roles["frontend"]]
    llm_device, speech_device = service_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES", ""), args.gpu_layout
    )
    os.environ["SOULX_MODEL_TOKEN"] = secrets.token_urlsafe(32)
    os.environ["SOULX_FRONTEND_MODEL"] = runner.model
    os.environ["SOULX_FRONTEND_BASE_URL"] = runner.base_url
    os.environ["SOULX_TOOLS_CONFIG"] = str(args.frontend_tools_override.resolve())
    os.environ["SOULX_VAD_URL"] = f"ws://127.0.0.1:{vad_port}/turn"
    os.environ["SOULX_TTS_URL"] = f"http://127.0.0.1:{tts_port}/tts"
    if runner.api_key_env:
        raise ValueError("Hosting requires an unauthenticated loopback frontend model runner")
    if not runner.base_url.startswith("http://127.0.0.1:"):
        raise ValueError("Hosting requires a loopback frontend model runner")
    soulx = args.source_dir.resolve() / "SoulX-Duplug"
    specs = [
        (
            "soulx-vad",
            [
                "env", "CUDA_VISIBLE_DEVICES=" + speech_device,
                "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(vad_port),
            ],
            soulx,
        ),
        (
            "soulx-tts",
            [
                "env", "CUDA_VISIBLE_DEVICES=" + speech_device,
                "PYTHONPATH=" + str(ROOT), "python", "-m", "scripts.serve_tts",
                "--host", "127.0.0.1", "--port", str(tts_port), "--model_dir",
                str(soulx / "pretrained_models/Index-TTS-1.5-vLLM"),
            ],
            soulx / "dialogue_system/modules/index_tts_vllm",
        ),
        (
            None,
            [
                "uvicorn", "soulx_agent.model_service:app", "--host", "127.0.0.1",
                "--port", str(args.port), "--no-access-log",
            ],
            ROOT,
        ),
    ]
    if runner.command:
        specs.insert(
            0, (runner.environment, ["env", "CUDA_VISIBLE_DEVICES=" + llm_device, *runner.command], ROOT)
        )
    processes = []

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        for environment, command, cwd in specs:
            processes.append(launch(environment, command, cwd))
        tunnel = subprocess.Popen(
            [args.cloudflared, "tunnel", "--url", f"http://127.0.0.1:{args.port}", "--no-autoupdate"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        processes.append(tunnel)
        # Drain Cloudflare diagnostics in a thread; never redirect Slurm logs.
        import queue
        import threading

        lines = queue.Queue()

        def read_tunnel():
            for line in tunnel.stdout:
                print(line, end="", flush=True)
                lines.put(line)

        threading.Thread(target=read_tunnel, daemon=True).start()
        published = False
        deadline = time.monotonic() + 120
        while True:
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"Hosting child exited with status {process.returncode}")
            if not published:
                if time.monotonic() > deadline:
                    raise TimeoutError("Cloudflare did not publish a temporary URL")
                try:
                    line = lines.get(timeout=1)
                except queue.Empty:
                    continue
                match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if not match:
                    continue
                url = match.group()
                script = (
                    '#!/bin/zsh\n'
                    'cd "${0:A:h}" || exit 1\n'
                    'source "$HOME/.zshrc"\n'
                    'export VECTOR_KEY\n'
                    f'export SOULX_MODEL_TOKEN={shlex.quote(os.environ["SOULX_MODEL_TOKEN"])}\n'
                    'exec "${SOULX_PYTHON:-python3}" -m scripts.run_desktop '
                    f'--frontend_url {shlex.quote(url)} "$@"\n'
                )
                launcher.write_text(script)
                launcher.chmod(0o700)
                bundle = args.output_dir / "laptop_client.tar.gz"
                with tarfile.open(bundle, "w:gz") as archive:
                    for name in ["soulx_agent", "scripts", "config", "desktop"]:
                        for path in sorted((ROOT / name).rglob("*")):
                            if path.is_file() and "__pycache__" not in path.parts and "node_modules" not in path.parts:
                                archive.add(path, arcname=str(path.relative_to(ROOT)))
                    for name in ["pyproject.toml", "README.md"]:
                        archive.add(ROOT / name, arcname=name)
                    archive.add(launcher, arcname="launch_laptop.sh")
                bundle.chmod(0o600)
                print(f"Frontend URL: {url}", flush=True)
                print(f"Private laptop bundle: {bundle}", flush=True)
                print("Models may still be loading. The desktop reports unavailable services until ready.", flush=True)
                published = True
            else:
                time.sleep(1)
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    main()
