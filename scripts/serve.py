import argparse
import os
from pathlib import Path

import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--frontend_runtime", choices=["soulx", "sglang_omni"], default="soulx")
    parser.add_argument("--omni_config", type=Path, default=Path("config/omni.yaml"))
    parser.add_argument("--frontend_tools_override", type=Path, default=None)
    args = parser.parse_args()
    if args.frontend_tools_override is not None:
        from soulx_agent.frontend_tools import load_override

        load_override(args.frontend_tools_override)
        os.environ["SOULX_FRONTEND_TOOLS_OVERRIDE"] = str(args.frontend_tools_override.resolve())
    else:
        os.environ.pop("SOULX_FRONTEND_TOOLS_OVERRIDE", None)
    os.environ["SOULX_FRONTEND_RUNTIME"] = args.frontend_runtime
    os.environ["SOULX_OMNI_CONFIG"] = str(args.omni_config.resolve())
    uvicorn.run("soulx_agent.server:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
