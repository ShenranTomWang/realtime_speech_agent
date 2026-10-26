import io
import json

import httpx
import numpy as np
import soundfile as sf

from soulx_agent import local_caller


async def test_local_caller_uses_only_local_audio_transcript_and_its_goal(monkeypatch):
    monkeypatch.setenv("SOULX_CALLER_BASE_URL", "http://127.0.0.1:8011/v1")
    monkeypatch.setenv("SOULX_CALLER_MODEL", "Qwen/Qwen3.5-4B")
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.host == "127.0.0.1"
        assert body["messages"][-1] == {
            "role": "user",
            "content": "What is your booking reference?",
        }
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"utterance":"ABC123", "end_call":false}'}}]
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        local_caller.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    caller = local_caller.LocalCaller(
        "Your booking reference is ABC123; ask to change your flight."
    )
    result = await caller.reply("What is your booking reference?")
    assert result == {"utterance": "ABC123", "end_call": False}
    assert len(caller.messages) == 3
    assert "ground_truth" not in json.dumps(requests)


def test_local_speech_conversion_returns_real_pcm_at_requested_rate():
    wav = io.BytesIO()
    sf.write(wav, np.zeros(2400, dtype=np.float32), 24000, format="WAV")
    audio = local_caller.decode_speech(wav.getvalue(), 16000)
    assert len(audio) == 1600 * 2


def test_tau_positional_instructions_preserve_chronology_and_input():
    from soulx_agent.local_caller import positional_caller_messages

    messages = [
        {"role": "system", "content": "Play the customer"},
        {"role": "user", "content": "How can I help?"},
        {"role": "system", "content": "You have 10 seconds left"},
    ]
    result = positional_caller_messages(messages)
    assert [message["role"] for message in result] == ["system", "user", "user"]
    assert result[-1]["content"] == "[Simulator instruction]\nYou have 10 seconds left"
    assert messages[-1]["role"] == "system"


def test_tau_combines_multiple_initial_system_instructions():
    from soulx_agent.local_caller import positional_caller_messages

    messages = [
        {"role": "system", "content": "Persona"},
        {"role": "system", "content": "Simulation rules"},
        {"role": "user", "content": "Hello"},
    ]
    result = positional_caller_messages(messages)
    assert result == [
        {"role": "system", "content": "Persona\n\nSimulation rules"},
        {"role": "user", "content": "Hello"},
    ]
