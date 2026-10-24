"""Fetch exact source revisions; explicitly opt in to weight downloads."""

import argparse
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--download_models", action="store_true")
    args = parser.parse_args()
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    args.source_dir.mkdir(parents=True, exist_ok=True)
    for name, item in lock.items():
        destination = args.source_dir / name
        if not destination.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    item["url"],
                    str(destination),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(destination), "checkout", "--detach", item["revision"]],
                check=True,
            )
        revision = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != item["revision"]:
            raise RuntimeError(f"Existing checkout differs from source lock: {destination}")
    soulx = args.source_dir / "SoulX-Duplug"
    config_path = soulx / "config/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["infer_config"]["asr"] = {"model_name": "sensevoice", "language": "en"}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if args.download_models:
        from huggingface_hub import snapshot_download

        snapshot_download("Qwen/Qwen3.5-4B")
        snapshot_download("Soul-AILab/SoulX-Duplug-0.6B", local_dir=soulx / "pretrained_models")
        from modelscope import snapshot_download as modelscope_download

        modelscope_download(
            "kusuriuri/Index-TTS-1.5-vLLM",
            local_dir=str(soulx / "pretrained_models/Index-TTS-1.5-vLLM"),
        )
    print("Pinned sources ready; SoulX ASR configured for English.")


if __name__ == "__main__":
    main()
