"""Live time and weather tools, independent of backend task state."""

from datetime import datetime, timezone

import httpx

from ..contracts import Tool


def build_tools(coordinator):
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
