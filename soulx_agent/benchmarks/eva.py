"""Retain EVA's audio capture, real ToolExecutor, caller and official metric engine."""

import os

import httpx
from openai import AsyncOpenAI


def install():
    if os.environ.get("SOULX_CALLER_MODE") == "local":
        from .local_eva import install as install_local

        install_local()
    if os.environ.get("SOULX_BENCHMARK_PHASE") == "inference":
        from .eva_inference import install as install_inference

        install_inference()
    elif os.environ.get("SOULX_BENCHMARK_PHASE") == "eval":
        from eva.metrics import runner as metrics_runner
        from eva.user_simulator.base import load_behavior_prompts

        resolve_user_config = metrics_runner.resolve_user_config
        default_persona = load_behavior_prompts()["default"]

        def resolve_release_user_config(*args, **kwargs):
            resolved = resolve_user_config(*args, **kwargs)
            resolved.setdefault("user_persona", default_persona)
            return resolved

        metrics_runner.resolve_user_config = resolve_release_user_config
    from eva.assistant import openai_realtime_server as module

    class SoulXAssistantServer(module.OpenAIRealtimeAssistantServer):
        _service_name = "SoulX + Claude Code"
        _metrics_processor_name = "soulx_claude_code"

        def _create_client(self):
            url = os.environ.get("SOULX_SERVER_URL", "ws://127.0.0.1:8765/v1/realtime")
            base = (
                url.replace("ws://", "http://")
                .replace("wss://", "https://")
                .removesuffix("/realtime")
            )
            return AsyncOpenAI(
                api_key="local", base_url=base, http_client=httpx.AsyncClient(trust_env=False)
            )

        async def _handle_openai_event(self, event, conn, websocket, audio_output_queue):
            if event.type == "session.updated":
                self.soulx_session_id = event.session.id
            if event.type == "error":
                raise RuntimeError(f"SoulX runtime reported an error: {event.error}")
            await super()._handle_openai_event(event, conn, websocket, audio_output_queue)

        async def _shutdown(self):
            await super()._shutdown()
            if getattr(self, "soulx_session_id", None):
                url = os.environ.get("SOULX_SERVER_URL", "ws://127.0.0.1:8765/v1/realtime")
                base = (
                    url.split("/v1/realtime")[0]
                    .replace("ws://", "http://")
                    .replace("wss://", "https://")
                )
                async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
                    response = await client.delete(base + "/sessions/" + self.soulx_session_id)
                    response.raise_for_status()

    module.OpenAIRealtimeAssistantServer = SoulXAssistantServer
