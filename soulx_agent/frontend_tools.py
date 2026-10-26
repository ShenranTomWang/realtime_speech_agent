"""Strict frontend tool selection; backend tools are never filtered here."""

from pathlib import Path

import yaml


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


def select_tools(environment, custom_tools, spec):
    available = {}
    for tool in environment + custom_tools:
        if tool.name in available:
            raise ValueError(f"Duplicate frontend tool name: {tool.name}")
        available[tool.name] = tool
    if spec is None:
        return list(available.values())
    unknown = set(spec["tools"]) - available.keys()
    if unknown:
        raise ValueError(f"Unknown frontend tools: {sorted(unknown)}")
    return [available[name] for name in spec["tools"]]
