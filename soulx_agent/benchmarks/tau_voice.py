"""Reuse Tau's continuous PCM24k transport; explicitly replace the Launa provider."""

import os
import uuid

import httpx


def install():
    if os.environ.get("SOULX_CALLER_MODE") == "local":
        from .local_tau import install as install_local

        install_local()
    import tau2.voice.audio_native.launa.discrete_time_adapter as adapter
    from tau2.voice.audio_native.launa.provider import LaunaRealtimeWSProvider

    class SoulXProvider(LaunaRealtimeWSProvider):
        def __init__(self, **kwargs):
            self.soulx_id = uuid.uuid4().hex
            endpoint = os.environ.get("SOULX_SERVER_URL", "ws://127.0.0.1:8765/v1/realtime")
            super().__init__(
                endpoint=endpoint, api_key="local", allow_insecure_local=True, **kwargs
            )
            # Validation is applied before adding the nonsecret session ID.
            self.endpoint += "?session_id=" + self.soulx_id

        async def disconnect(self):
            await super().disconnect()
            base = (
                self.endpoint.split("/v1/realtime")[0]
                .replace("ws://", "http://")
                .replace("wss://", "https://")
            )
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.delete(base + "/sessions/" + self.soulx_id)
                response.raise_for_status()

    adapter.LaunaRealtimeWSProvider = SoulXProvider
