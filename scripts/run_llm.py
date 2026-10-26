"""Host the selected inference runners and API adapter independently of voice agents."""

import argparse
import asyncio
import os
import secrets
import signal
import time
from pathlib import Path

import httpx

from soulx_agent.llm_runner import DEFAULT_CONFIG, ROOT, load_llm_plan
from soulx_agent.processes import launch, stop_processes


async def ready(plan, processes):
    timeout = float(os.environ.get("SOULX_STARTUP_TIMEOUT", "1800"))
    await plan.wait_for_models(processes, timeout=timeout)
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("An LLM service exited during startup")
            try:
                response = await client.get(plan.gateway_url + "/health/liveliness")
                response.raise_for_status()
                return
            except httpx.HTTPError:
                await asyncio.sleep(2)
    raise TimeoutError("LLM gateway did not become ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_config", type=Path, default=os.environ.get("SOULX_LLM_CONFIG", DEFAULT_CONFIG)
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    plan = load_llm_plan(args.llm_config)
    if not os.environ.get("SOULX_GATEWAY_KEY"):
        os.environ["SOULX_GATEWAY_KEY"] = "sk-" + secrets.token_urlsafe(32)
    args.output_dir = args.output_dir.resolve()
    gateway_config = args.output_dir / "gateway.yaml"
    (args.output_dir / "ready.json").unlink(missing_ok=True)
    plan.write_gateway_config(gateway_config)
    processes = []

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        for runner in plan.selected_runners:
            if runner.command:
                processes.append(launch(runner.environment, list(runner.command), ROOT))
        processes.append(
            launch(
                plan.gateway_environment,
                [
                    "litellm",
                    "--config",
                    str(gateway_config),
                    "--host",
                    plan.gateway_host,
                    "--port",
                    str(plan.gateway_port),
                ],
                ROOT,
            )
        )
        asyncio.run(ready(plan, processes))
        (args.output_dir / "ready.json").write_text('{"ready": true}\n')
        while True:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("An LLM service exited")
            time.sleep(1)
    finally:
        (args.output_dir / "ready.json").unlink(missing_ok=True)
        stop_processes(processes)


if __name__ == "__main__":
    main()
