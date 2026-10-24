"""Launch upstream IndexTTS without its shared torch.compile artifact cache.

This affects only the speech process. The separate SGLang LLM keeps its normal
execution mode. Eager execution avoids concurrent NFS compiler-cache writes.
"""

import runpy
import sys
from pathlib import Path


def main():
    import vllm.engine.arg_utils as engine_args

    original = engine_args.AsyncEngineArgs

    class EagerSpeechArgs(original):
        def __init__(self, *args, **kwargs):
            kwargs["enforce_eager"] = True
            super().__init__(*args, **kwargs)

    engine_args.AsyncEngineArgs = EagerSpeechArgs
    server = Path.cwd() / "api_server.py"
    if not server.is_file():
        raise FileNotFoundError("Run the speech launcher from the upstream IndexTTS directory")
    sys.path.insert(0, str(server.parent))
    runpy.run_path(str(server), run_name="__main__")


if __name__ == "__main__":
    main()
