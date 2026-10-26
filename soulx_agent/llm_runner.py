"""Model hosting/routing only; this layer has no agent prompts, tools or harness imports."""

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/llm.yaml"
ROLES = ("frontend", "backend")


def _base_url(value):
    if not isinstance(value, str):
        raise ValueError("LLM base_url must be a string")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("LLM base_url must be an HTTP(S) endpoint without embedded credentials")
    return value.rstrip("/")


@dataclass(frozen=True)
class ModelRunner:
    name: str
    model: str
    base_url: str
    api_key_env: str | None
    environment: str | None
    command: tuple[str, ...]
    chat_template_kwargs: dict = field(default_factory=dict)

    def validate_credentials(self):
        if self.api_key_env and not os.environ.get(self.api_key_env):
            raise RuntimeError(
                f"Runner {self.name} requires environment variable {self.api_key_env}"
            )

    def gateway_parameters(self):
        parameters = {
            "model": "openai/" + self.model,
            "api_base": self.base_url,
            "api_key": "os.environ/" + self.api_key_env if self.api_key_env else "local",
        }

        if self.chat_template_kwargs:
            parameters["extra_body"] = {"chat_template_kwargs": self.chat_template_kwargs}
        return parameters

    def public_config(self):
        return {
            "model": self.model,
            "chat_template_kwargs": self.chat_template_kwargs,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "launch": {"environment": self.environment, "command": list(self.command)}
            if self.command
            else None,
        }


@dataclass(frozen=True)
class LLMPlan:
    roles: dict[str, str]
    runners: dict[str, ModelRunner]
    gateway_environment: str | None = "soulx-gateway"
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 4000

    @property
    def gateway_url(self):
        return f"http://{self.gateway_host}:{self.gateway_port}"

    @property
    def selected_runners(self):
        return [self.runners[name] for name in dict.fromkeys(self.roles.values())]

    def validate_credentials(self):
        if not os.environ.get("SOULX_GATEWAY_KEY"):
            raise RuntimeError("Required environment variable is not set: SOULX_GATEWAY_KEY")
        for runner in self.selected_runners:
            runner.validate_credentials()

    def gateway_config(self):
        return {
            "model_list": [
                {
                    "model_name": role,
                    "litellm_params": self.runners[self.roles[role]].gateway_parameters(),
                }
                for role in ROLES
            ],
            "litellm_settings": {
                "drop_params": True,
                "use_chat_completions_url_for_anthropic_messages": True,
            },
            "general_settings": {"master_key": "os.environ/SOULX_GATEWAY_KEY"},
        }

    def public_config(self):
        return {
            "roles": self.roles,
            "runners": {runner.name: runner.public_config() for runner in self.selected_runners},
            "gateway": {
                "environment": self.gateway_environment,
                "host": self.gateway_host,
                "port": self.gateway_port,
            },
        }

    def write_gateway_config(self, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(self.gateway_config(), sort_keys=False))

    async def warmup_managed_models(self, timeout=1800):
        # A model-list endpoint may respond before CUDA kernels are compiled.
        for runner in self.selected_runners:
            if not runner.command:
                continue
            print(f"Warming model generation: {runner.name}", flush=True)
            headers = {}
            if runner.api_key_env:
                headers["Authorization"] = "Bearer " + os.environ[runner.api_key_env]
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                response = await client.post(
                    runner.base_url + "/chat/completions",
                    headers=headers,
                    json={
                        "model": runner.model,
                        "messages": [{"role": "user", "content": "Say hello in one word."}],
                        "max_tokens": 32,
                        "chat_template_kwargs": runner.chat_template_kwargs,
                    },
                )
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                if not message.get("content"):
                    raise RuntimeError(f"Runner {runner.name} warmup returned no text")
            print(f"Model generation ready: {runner.name}", flush=True)

    async def wait_for_models(self, processes, timeout=600):
        pending = {runner.name: runner for runner in self.selected_runners}
        deadline = time.monotonic() + timeout
        last_report = 0
        while pending and time.monotonic() < deadline:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("An LLM service exited during startup")
            for name, runner in list(pending.items()):
                headers = {}
                if runner.api_key_env:
                    headers["Authorization"] = "Bearer " + os.environ[runner.api_key_env]
                loopback = urlsplit(runner.base_url).hostname in {"localhost", "127.0.0.1", "::1"}
                try:
                    async with httpx.AsyncClient(timeout=5, trust_env=not loopback) as client:
                        response = await client.get(runner.base_url + "/models", headers=headers)
                        response.raise_for_status()
                        data = response.json()
                        model_ids = {item["id"] for item in data["data"]}
                        if runner.model not in model_ids:
                            raise ValueError(
                                f"Runner {name} does not serve configured model {runner.model}"
                            )
                    pending.pop(name)
                except httpx.HTTPStatusError as error:
                    if error.response.status_code in {401, 403, 404}:
                        raise RuntimeError(
                            f"Runner {name} rejected its model-list readiness request"
                        ) from error
                except (httpx.RequestError, KeyError, ValueError) as error:
                    # Connection/model startup may take minutes for a managed local server.
                    if isinstance(error, ValueError) and not isinstance(error, KeyError):
                        raise
            if pending:
                if time.monotonic() - last_report >= 30:
                    print("LLM startup waiting for: " + ", ".join(pending), flush=True)
                    last_report = time.monotonic()
                await asyncio.sleep(2)
        if pending:
            raise TimeoutError("LLM readiness timed out for: " + ", ".join(pending))


def load_llm_plan(path: Path | str = DEFAULT_CONFIG):
    document = yaml.safe_load(Path(path).read_text())
    port_offset = int(os.environ.get("SOULX_PORT_OFFSET", "0"))
    if not 0 <= port_offset <= 20000:
        raise ValueError("SOULX_PORT_OFFSET must be between 0 and 20000")
    if not isinstance(document, dict) or set(document) - {"roles", "runners", "gateway"}:
        raise ValueError("LLM config must define roles, runners and optional gateway settings")
    roles = document.get("roles")
    definitions = document.get("runners")
    if not isinstance(roles, dict) or set(roles) != set(ROLES):
        raise ValueError("LLM roles must define exactly frontend and backend")
    if not isinstance(definitions, dict):
        raise ValueError("LLM runners must be a mapping")
    runners = {}
    for name, definition in definitions.items():
        if not isinstance(definition, dict) or set(definition) - {
            "model",
            "base_url",
            "api_key_env",
            "launch",
            "chat_template_kwargs",
        }:
            raise ValueError(f"Invalid LLM runner definition: {name}")
        if not isinstance(definition.get("model"), str) or not definition["model"].strip():
            raise ValueError(f"Runner {name} requires a model ID")
        api_key_env = definition.get("api_key_env")
        if api_key_env is not None and (
            not isinstance(api_key_env, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env)
        ):
            raise ValueError(f"Runner {name} api_key_env must name an environment variable")
        launch = definition.get("launch") or {}
        if not isinstance(launch, dict) or set(launch) - {"environment", "command"}:
            raise ValueError(f"Runner {name} launch requires environment and command")
        command = launch.get("command", [])
        if not isinstance(command, list) or any(
            not isinstance(arg, str) or not arg for arg in command
        ):
            raise ValueError(f"Runner {name} command must be an argument list")
        if launch and not command:
            raise ValueError(f"Runner {name} launch requires a nonempty command")
        environment = launch.get("environment")
        if environment is not None and (
            not isinstance(environment, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", environment)
        ):
            raise ValueError(
                f"Runner {name} environment must be a scratch environment name or null"
            )
        base_url = _base_url(definition.get("base_url"))
        if command and port_offset:
            parsed = urlsplit(base_url)
            if parsed.hostname not in {"localhost", "127.0.0.1"} or parsed.port is None:
                raise ValueError(
                    "Managed runners with a port offset require a loopback URL with an explicit port"
                )
            base_url = urlunsplit(
                parsed._replace(netloc=f"{parsed.hostname}:{parsed.port + port_offset}")
            )
            command = list(command)
            if "--port" not in command:
                raise ValueError(
                    "Managed runner command requires --port for concurrent benchmark jobs"
                )
            position = command.index("--port") + 1
            command[position] = str(int(command[position]) + port_offset)
        chat_template_kwargs = definition.get("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a mapping")
        runners[name] = ModelRunner(
            name,
            definition["model"],
            base_url,
            api_key_env,
            environment,
            tuple(command),
            chat_template_kwargs,
        )
    if any(not isinstance(value, str) or value not in runners for value in roles.values()):
        raise ValueError("Every role must reference a configured LLM runner")
    gateway = document.get("gateway", {})
    if not isinstance(gateway, dict) or set(gateway) - {"environment", "host", "port"}:
        raise ValueError("Invalid gateway settings")
    host = gateway.get("host", "127.0.0.1")
    port = gateway.get("port", 4000) + port_offset
    if host not in {"127.0.0.1", "localhost"} or type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Gateway requires a loopback host and valid port")
    environment = gateway.get("environment", "soulx-gateway")
    if environment is not None and (
        not isinstance(environment, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", environment)
    ):
        raise ValueError("Gateway environment must be a scratch environment name or null")
    return LLMPlan(dict(roles), runners, environment, host, port)
