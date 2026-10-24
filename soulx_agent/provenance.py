"""Reject accidental resume with changed runtime code, models or source revisions."""

import hashlib
import json
import os
from pathlib import Path

from .frontend_tools import load_override
from .llm_runner import DEFAULT_CONFIG, load_llm_plan

ROOT = Path(__file__).resolve().parents[1]


def verify_run(output_dir: Path, source_dir: Path, benchmark: str, llm_config=DEFAULT_CONFIG):
    paths = [ROOT / "sources.lock.json", ROOT / "uv.lock"]
    for folder in ["soulx_agent", "scripts", "integrations", "web"]:
        paths.extend(
            path
            for path in (ROOT / folder).rglob("*")
            if path.suffix in {".py", ".sh", ".js", ".html", ".css"}
        )
    frontend_runtime = os.environ.get("SOULX_FRONTEND_RUNTIME", "soulx")
    omni_config = Path(os.environ.get("SOULX_OMNI_CONFIG", ROOT / "config/omni.yaml"))
    omni_hash = None
    if frontend_runtime == "sglang_omni":
        omni_hash = hashlib.sha256(omni_config.read_bytes()).hexdigest()
    paths = sorted(paths)
    fingerprint = hashlib.sha256()
    for path in paths:
        fingerprint.update(str(path.relative_to(ROOT)).encode())
        fingerprint.update(path.read_bytes())
    detector = source_dir / "SoulX-Duplug/config/config.yaml"
    manifest = {
        "schema_version": 4,
        "frontend_tools_override": load_override(os.environ.get("SOULX_FRONTEND_TOOLS_OVERRIDE")),
        "frontend_runtime": frontend_runtime,
        "omni_config_sha256": omni_hash,
        "llm_runtime": load_llm_plan(llm_config).public_config(),
        "benchmark": benchmark,
        "caller": {
            "mode": os.environ.get("SOULX_CALLER_MODE", "local"),
            "llm_role": "frontend",
            "tts": "IndexTTS",
            "asr": "SoulX SenseVoice",
            "speaker": os.environ.get("SOULX_CALLER_SPEAKER", "elva"),
            "scoring": "deferred; Tau inference records environment checks only",
        }
        if benchmark in {"tau_voice", "eva"}
        else None,
        "implementation_sha256": fingerprint.hexdigest(),
        "source_lock": json.loads((ROOT / "sources.lock.json").read_text()),
        "detector_config_sha256": hashlib.sha256(detector.read_bytes()).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "run_manifest.json"
    if destination.exists():
        if json.loads(destination.read_text()) != manifest:
            raise ValueError(
                "Runtime/model/source configuration changed: use a new output directory"
            )
    else:
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(destination)
