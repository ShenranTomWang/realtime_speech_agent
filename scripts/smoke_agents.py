"""Exercise real model responses and MCP execution through both Claude Code agents."""

import argparse
import asyncio
import json
import secrets
import time
from pathlib import Path

import httpx

from soulx_agent.contracts import ModelConnection, Tool
from soulx_agent.harness import ClaudeHarness
from soulx_agent.llm_runner import load_llm_plan


async def check_non_thinking(frontend):
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        response = await client.post(
            frontend.base_url + "/chat/completions",
            json={
                "model": frontend.model,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 32,
                "chat_template_kwargs": frontend.chat_template_kwargs or {"enable_thinking": False},
            },
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        if message.get("reasoning") or message.get("reasoning_content"):
            raise RuntimeError("Frontend produced reasoning despite thinking being disabled")
        if not message.get("content") or "<think>" in message["content"]:
            raise RuntimeError("Frontend did not produce a normal non-thinking reply")


async def check(plan, output_dir):
    frontend = plan.runners[plan.roles["frontend"]]
    if (
        '{"enable_thinking":false}' in frontend.command
        or frontend.chat_template_kwargs.get("enable_thinking") is False
    ):
        await check_non_thinking(frontend)
    results = {}
    for role in ("frontend", "backend"):
        calls = []
        nonce = secrets.token_hex(8)

        async def lookup(arguments):
            calls.append(arguments)
            return {"confirmation": nonce}

        harness = ClaudeHarness(
            ModelConnection(role, plan.gateway_url, "SOULX_GATEWAY_KEY"),
            "Use the lookup tool to obtain the confirmation. Reply with its value only.",
            [
                Tool(
                    "lookup",
                    "Return a confirmation value.",
                    {"type": "object", "properties": {}},
                    lookup,
                )
            ],
        )
        started = time.monotonic()
        answer = await asyncio.wait_for(harness.run("Obtain the confirmation now."), timeout=180)
        if not calls or nonce not in answer:
            raise RuntimeError(f"{role} failed to execute its tool and report its result")
        results[role] = {"tool_calls": len(calls), "elapsed_seconds": time.monotonic() - started}
        print(f"{role}: real Claude Code tool round trip passed", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "agent_smoke.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    plan = load_llm_plan(args.llm_config)
    plan.validate_credentials()
    asyncio.run(check(plan, args.output_dir))


if __name__ == "__main__":
    main()
