"""Discover trusted local tool providers and bind their tools to a coordinator.

Provider modules expose build_tools(coordinator), returning a list of Tool objects.
Factories must only bind callbacks: coordinator may be None when listing names.
"""

from importlib import import_module
from pkgutil import iter_modules


def load_tools(coordinator=None):
    tools = {}
    for entry in sorted(iter_modules(__path__), key=lambda entry: entry.name):
        if entry.name.startswith("_"):
            continue
        provider = import_module(f"{__name__}.{entry.name}")
        for tool in provider.build_tools(coordinator):
            if tool.name in tools:
                raise ValueError(f"Duplicate toolchain tool name: {tool.name}")
            tools[tool.name] = tool
    return list(tools.values())


def tool_names():
    return {tool.name for tool in load_tools()}
