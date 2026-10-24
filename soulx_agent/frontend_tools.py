"""Strict frontend tool selection; backend tools are never filtered here."""

from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from .contracts import Tool


def load_override(path):
    if path is None:
        return None
    spec = yaml.safe_load(Path(path).read_text())
    if not isinstance(spec, dict) or set(spec) - {"tools", "instructions"}:
        raise ValueError("Frontend override must contain tools and optional instructions")
    names = spec.get("tools")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("Frontend override tools must be a list of exact tool names")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate frontend override tool names")
    if not isinstance(spec.get("instructions", ""), str):
        raise ValueError("Frontend override instructions must be a string")
    return spec


def simple_tools():
    async def date(args):
        return {"utc": datetime.now(timezone.utc).isoformat(), "clock": "live wall clock"}

    async def weather(args):
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": args["latitude"],
                    "longitude": args["longitude"],
                    "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
                    "timezone": "UTC",
                },
            )
            response.raise_for_status()
            return {"source": "Open-Meteo", "clock": "live weather", **response.json()}

    return [
        Tool(
            "get_current_datetime",
            "Get LIVE UTC wall-clock time. Never replace a benchmark/scenario reference date with this.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            date,
        ),
        Tool(
            "get_weather",
            "Get LIVE current weather at known coordinates from Open-Meteo. Never substitute for scenario weather.",
            {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                    "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            weather,
        ),
    ]


def select_tools(environment, task_tools, spec):
    available = {tool.name: tool for tool in environment + task_tools}
    # Add helpers to every profile so tool availability is the only ablation.
    for tool in simple_tools():
        available.setdefault(tool.name, tool)
    if spec is None:
        return list(available.values())
    unknown = set(spec["tools"]) - available.keys()
    if unknown:
        raise ValueError(f"Unknown frontend tools: {sorted(unknown)}")
    return [available[name] for name in spec["tools"]]
