from types import SimpleNamespace

import pytest

from scripts import run_stack


async def test_startup_timeout_identifies_missing_llm_marker(tmp_path, monkeypatch):
    times = iter((0, 2))
    monkeypatch.setattr(run_stack, "time", SimpleNamespace(monotonic=lambda: next(times)))
    monkeypatch.setenv("SOULX_STARTUP_TIMEOUT", "1")
    with pytest.raises(TimeoutError, match="1s; waiting for LLM readiness marker"):
        await run_stack.wait_services([], "http://127.0.0.1:4000", tmp_path / "ready.json")
