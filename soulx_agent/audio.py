"""SoulX's native float32/16 kHz turn socket and its IndexTTS WAV service."""

import base64
import io
import json
import os
import uuid

import httpx
import numpy as np
import soundfile as sf
import soxr
import websockets


class SoulXAudio:
    def __init__(self, input_rate=24000, token=None):
        self.input_rate = input_rate
        self.token = token
        self.resampler = soxr.ResampleStream(input_rate, 16000, 1, dtype="float32")
        self.ws = None
        self.session_id = uuid.uuid4().hex

    async def connect(self):
        self.ws = await websockets.connect(
            os.environ.get("SOULX_VAD_URL", "ws://127.0.0.1:8000/turn"),
            additional_headers=(
                {"Authorization": "Bearer " + self.token} if self.token else None
            ),
        )

    async def process(self, pcm):
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        samples = self.resampler.resample_chunk(samples)
        if not len(samples):
            return {"state": "blank"}
        await self.ws.send(
            json.dumps(
                {
                    "type": "audio",
                    "session_id": self.session_id,
                    "audio": base64.b64encode(samples.astype("<f4").tobytes()).decode(),
                }
            )
        )
        response = json.loads(await self.ws.recv())
        if "state" not in response:
            raise RuntimeError("SoulX returned no turn state")
        return response["state"]

    async def close(self):
        if self.ws:
            await self.ws.close()


async def synthesize(text, token=None):
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        response = await client.post(
            os.environ.get("SOULX_TTS_URL", "http://127.0.0.1:6006/tts"),
            json={"text": text, "character": os.environ.get("SOULX_SPEAKER", "elva")},
            headers=(
                {"Authorization": "Bearer " + token} if token else None
            ),
        )
        response.raise_for_status()
    audio, rate = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 24000:
        audio = soxr.resample(audio, rate, 24000)
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
