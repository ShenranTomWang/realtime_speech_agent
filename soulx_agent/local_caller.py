"""Local caller language and speech services, with isolated conversation context."""

import io
import json
import os

import httpx
import numpy as np
import soundfile as sf
import soxr


def decode_speech(wav, sample_rate):
    audio, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != sample_rate:
        audio = soxr.resample(audio, rate, sample_rate)
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def speech_request(text):
    return {
        "text": text,
        "character": os.environ.get("SOULX_CALLER_SPEAKER", "elva"),
    }


def synthesize_caller_sync(text, sample_rate=24000):
    with httpx.Client(timeout=120, trust_env=False) as client:
        response = client.post(
            os.environ.get("SOULX_TTS_URL", "http://127.0.0.1:6006/tts"),
            json=speech_request(text),
        )
        response.raise_for_status()
    return decode_speech(response.content, sample_rate)


async def synthesize_caller(text, sample_rate=16000):
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        response = await client.post(
            os.environ.get("SOULX_TTS_URL", "http://127.0.0.1:6006/tts"),
            json=speech_request(text),
        )
        response.raise_for_status()
    return decode_speech(response.content, sample_rate)


class LocalCaller:
    def __init__(self, system_prompt):
        self.messages = [
            {
                "role": "system",
                "content": system_prompt
                + "\nReturn JSON with utterance (spoken words) and end_call (boolean). "
                "Play only the user. React to what you hear; do not invent assistant replies. "
                "Set end_call only after you have said goodbye, the agent has transferred you, "
                "or the conversation is resolved or cannot progress. Never end on the same turn "
                "that you provide requested information or approval. Use short natural speech.",
            }
        ]

    async def reply(self, heard):
        self.messages.append({"role": "user", "content": heard})
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            response = await client.post(
                os.environ["SOULX_CALLER_BASE_URL"].rstrip("/") + "/chat/completions",
                json={
                    "model": os.environ["SOULX_CALLER_MODEL"],
                    "messages": self.messages,
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "response_format": {"type": "json_object"},
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        result = json.loads(content)
        if not isinstance(result.get("utterance"), str) or not isinstance(
            result.get("end_call"), bool
        ):
            raise ValueError("Local caller returned invalid utterance/end_call JSON")
        self.messages.append({"role": "assistant", "content": content})
        return result


def positional_caller_messages(messages):
    """Keep later simulator instructions in place for strict chat templates."""
    result = []
    conversation_started = False
    for message in messages:
        if message.get("role") == "system" and conversation_started:
            message = {
                **message,
                "role": "user",
                "content": "[Simulator instruction]\n" + message["content"],
            }
        elif message.get("role") == "system" and result:
            result[0] = {
                **result[0],
                "content": result[0]["content"] + "\n\n" + message["content"],
            }
            continue
        elif message.get("role") != "system":
            conversation_started = True
        result.append(message)
    return result
