import json

import httpx
import pytest
import yaml

from soulx_agent import llm_runner
from soulx_agent.llm_runner import ROOT, load_llm_plan
from soulx_agent.provenance import verify_run


@pytest.mark.parametrize(
    "config, frontend_url, backend_url",
    [
        ("llm.yaml", "https://proxy.vectorinstitute.ai/v1", "https://proxy.vectorinstitute.ai/v1"),
        (
            "llm/local_frontend.yaml",
            "http://127.0.0.1:8011/v1",
            "https://proxy.vectorinstitute.ai/v1",
        ),
        (
            "llm/local_backend.yaml",
            "https://proxy.vectorinstitute.ai/v1",
            "http://127.0.0.1:8012/v1",
        ),
        ("llm/local.yaml", "http://127.0.0.1:8011/v1", "http://127.0.0.1:8012/v1"),
    ],
)
def test_independent_role_routes(config, frontend_url, backend_url):
    plan = load_llm_plan(ROOT / "config" / config)
    routes = {
        item["model_name"]: item["litellm_params"] for item in plan.gateway_config()["model_list"]
    }
    assert routes["frontend"]["api_base"] == frontend_url
    assert routes["backend"]["api_base"] == backend_url
    assert routes["frontend"]["model"].startswith("openai/")
    assert routes["backend"]["model"].startswith("openai/")


def test_all_local_requires_no_vector_key(monkeypatch):
    monkeypatch.delenv("VECTOR_KEY", raising=False)
    monkeypatch.setenv("SOULX_GATEWAY_KEY", "sk-local-only")
    plan = load_llm_plan(ROOT / "config/llm/local.yaml")
    plan.validate_credentials()
    assert all(runner.api_key_env is None for runner in plan.selected_runners)
    assert "VECTOR_KEY" not in json.dumps(plan.gateway_config())
    with pytest.raises(RuntimeError, match="VECTOR_KEY"):
        load_llm_plan(ROOT / "config/llm/local_frontend.yaml").validate_credentials()


def test_only_selected_local_server_is_started():
    plan = load_llm_plan(ROOT / "config/llm/managed_local.yaml")
    assert len(plan.selected_runners) == 1
    assert plan.roles["frontend"] == plan.roles["backend"]
    assert plan.selected_runners[0].command[:2] == ("vllm", "serve")
    external = load_llm_plan(ROOT / "config/llm/local.yaml")
    assert not any(runner.command for runner in external.selected_runners)


def test_gateway_config_never_embeds_credentials(tmp_path, monkeypatch):
    secret = "sk-test-not-a-real-credential"
    monkeypatch.setenv("VECTOR_KEY", secret)
    plan = load_llm_plan()
    destination = tmp_path / "gateway.yaml"
    plan.write_gateway_config(destination)
    assert secret not in destination.read_text()
    assert "os.environ/VECTOR_KEY" in destination.read_text()
    assert secret not in json.dumps(plan.public_config())


async def test_local_readiness_checks_both_models_without_remote_calls(monkeypatch):
    monkeypatch.delenv("VECTOR_KEY", raising=False)
    plan = load_llm_plan(ROOT / "config/llm/local.yaml")
    requests = []

    def respond(request):
        requests.append(request)
        assert request.url.host == "127.0.0.1"
        assert request.url.path == "/v1/models"
        assert "authorization" not in request.headers
        model = "frontend-local" if request.url.port == 8011 else "backend-local"
        return httpx.Response(200, json={"data": [{"id": model}]})

    original = httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        llm_runner.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs)
    )
    await plan.wait_for_models([], timeout=1)
    assert {request.url.port for request in requests} == {8011, 8012}


async def test_wrong_local_model_fails_readiness(monkeypatch):
    plan = load_llm_plan(ROOT / "config/llm/managed_local.yaml")
    original = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": [{"id": "wrong-model"}]})
    )
    monkeypatch.setattr(
        llm_runner.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs)
    )
    with pytest.raises(ValueError, match="does not serve"):
        await plan.wait_for_models([], timeout=1)


def test_resuming_with_changed_host_is_rejected(tmp_path):
    source = tmp_path / "sources"
    detector = source / "SoulX-Duplug/config/config.yaml"
    detector.parent.mkdir(parents=True)
    detector.write_text("infer_config: {}")
    output = tmp_path / "output"
    verify_run(output, source, "fdb_v3", ROOT / "config/llm.yaml")
    with pytest.raises(ValueError, match="configuration changed"):
        verify_run(output, source, "fdb_v3", ROOT / "config/llm/local_backend.yaml")


@pytest.mark.parametrize(
    "change",
    [
        {"roles": {"frontend": "missing", "backend": "vector"}},
        {"roles": {"frontend": "vector"}},
        {
            "runners": {
                "vector": {"model": "test", "base_url": "https://user:secret@example.com/v1"}
            }
        },
        {
            "runners": {
                "vector": {
                    "model": "test",
                    "base_url": "http://localhost:8011/v1",
                    "api_key": "inline-secret",
                }
            }
        },
    ],
)
def test_invalid_runner_config_is_rejected(tmp_path, change):
    document = yaml.safe_load((ROOT / "config/llm.yaml").read_text())
    document.update(change)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        load_llm_plan(path)


def test_benchmark_ports_do_not_collide_or_change_vector_host(monkeypatch):
    gateways = set()
    frontend_ports = set()
    for offset in (0, 100, 200):
        monkeypatch.setenv("SOULX_PORT_OFFSET", str(offset))
        plan = load_llm_plan(ROOT / "config/llm/qwen35_4b_sglang_vector.yaml")
        gateways.add(plan.gateway_url)
        runner = plan.runners[plan.roles["frontend"]]
        port = runner.command[runner.command.index("--port") + 1]
        assert runner.base_url == f"http://127.0.0.1:{port}/v1"
        frontend_ports.add(port)
        backend = plan.runners[plan.roles["backend"]]
        assert backend.base_url == "https://proxy.vectorinstitute.ai/v1"
    assert len(gateways) == len(frontend_ports) == 3


async def test_warmup_exercises_generation_only_for_managed_models(monkeypatch):
    plan = load_llm_plan(ROOT / "config/llm/qwen35_4b_sglang_vector.yaml")
    requests = []

    def respond(request):
        requests.append(request)
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        llm_runner.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    await plan.warmup_managed_models(timeout=1)
    assert len(requests) == 1
    assert requests[0].url.host == "127.0.0.1"
