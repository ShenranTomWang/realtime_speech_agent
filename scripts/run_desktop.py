"""Run both agent harnesses on macOS and connect to remote frontend models."""

import argparse
import os
import secrets
import signal
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import yaml

from soulx_agent.frontend_tools import load_override
from soulx_agent.processes import stop_processes

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend_url", required=True)
    parser.add_argument("--frontend_tools_override", type=Path)
    parser.add_argument("--working_dir", type=Path, default=Path.cwd())
    parser.add_argument("--output_dir", type=Path, default=Path.home() / ".soulx-agent")
    parser.add_argument("--port", type=int, default=9075)
    parser.add_argument("--gateway_port", type=int, default=4075)
    args = parser.parse_args()
    url = args.frontend_url.rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
        parser.error("--frontend_url must be an HTTPS URL without credentials, query or fragment")
    if args.port == args.gateway_port or not all(1 <= port <= 65535 for port in [args.port, args.gateway_port]):
        parser.error("Use two different valid ports")
    token = os.environ["SOULX_MODEL_TOKEN"]
    electron = ROOT / "desktop/node_modules/.bin/electron"
    if not electron.exists():
        parser.error(f"Install desktop dependencies first: cd {ROOT / 'desktop'} && npm install")
    # Fetch actual hosting configuration, not a credential or model startup probe.
    response = httpx.get(
        url + "/configuration", headers={"Authorization": "Bearer " + token}, timeout=30
    )
    response.raise_for_status()
    remote = response.json()
    run_dir = args.output_dir.expanduser() / secrets.token_hex(8)
    run_dir.mkdir(parents=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    profile = args.frontend_tools_override
    if profile is None:
        profile = run_dir / "frontend_tools.yaml"
        profile.write_text(yaml.safe_dump(remote["frontend_tools"]))
    load_override(profile)
    gateway_key = "sk-" + secrets.token_urlsafe(32)
    config = {
        "model_list": [
            {
                "model_name": "frontend",
                "litellm_params": {
                    "model": "openai/" + remote["frontend_model"],
                    "api_base": url + "/llm/v1", "api_key": "os.environ/SOULX_MODEL_TOKEN",
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                },
            },
            {
                "model_name": "backend",
                "litellm_params": {
                    "model": "openai/Qwen3_8-27B", "api_base": "https://proxy.vectorinstitute.ai/v1",
                    "api_key": "os.environ/VECTOR_KEY",
                },
            },
        ],
        "litellm_settings": {
            "drop_params": True, "use_chat_completions_url_for_anthropic_messages": True,
        },
        "general_settings": {"master_key": "os.environ/SOULX_GATEWAY_KEY"},
    }
    gateway_config = run_dir / "gateway.yaml"
    gateway_config.write_text(yaml.safe_dump(config))
    origin = f"http://127.0.0.1:{args.port}"
    os.environ.update(
        SOULX_GATEWAY_URL=f"http://127.0.0.1:{args.gateway_port}",
        SOULX_GATEWAY_KEY=gateway_key, SOULX_DESKTOP_KEY=secrets.token_urlsafe(32),
        SOULX_DESKTOP_ORIGIN=origin, SOULX_WORKING_DIR=str(args.working_dir.expanduser().resolve()),
        SOULX_FRONTEND_TOOLS_OVERRIDE=str(profile.resolve()), SOULX_RUN_DIR=str(run_dir / "sessions"),
        SOULX_VAD_URL="wss://" + parsed.netloc + parsed.path.rstrip("/") + "/turn",
        SOULX_TTS_URL=url + "/tts", SOULX_FRONTEND_RUNTIME="soulx",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
    )
    # Local gateway owns provider credentials; they are never sent to the Electron renderer.
    processes = []

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        processes.append(subprocess.Popen(
            [
                "litellm", "--config", str(gateway_config), "--host", "127.0.0.1",
                "--port", str(args.gateway_port),
            ], cwd=ROOT, start_new_session=True,
        ))
        import sys

        processes.append(subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "soulx_agent.desktop_runtime:app",
                "--host", "127.0.0.1", "--port", str(args.port), "--no-access-log",
            ], cwd=ROOT, start_new_session=True,
        ))
        # Electron retries navigation while the local HTTP process binds its socket.
        ui_env = {
            key: value for key, value in os.environ.items()
            if key not in {"VECTOR_KEY", "SOULX_MODEL_TOKEN", "SOULX_GATEWAY_KEY"}
        }
        ui_env["SOULX_UI_URL"] = origin
        desktop = subprocess.Popen(
            [str(electron), str(ROOT / "desktop")], cwd=ROOT, env=ui_env, start_new_session=True
        )
        processes.append(desktop)
        while desktop.poll() is None:
            for process in processes[:-1]:
                if process.poll() is not None:
                    raise RuntimeError(f"Desktop service exited: {process.returncode}")
            time.sleep(0.5)
        if desktop.returncode:
            raise RuntimeError(f"Desktop window exited: {desktop.returncode}")
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    main()
