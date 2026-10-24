"""Validate actual upstream schemas and local-caller adapters without running models."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["tau_voice", "eva"], required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["SOULX_CALLER_MODE"] = "local"
    os.environ["SOULX_BENCHMARK_PHASE"] = "inference"
    if args.benchmark == "tau_voice":
        os.environ["TAU2_DATA_DIR"] = str(args.dataset_dir / "tau_voice")
        from soulx_agent.benchmarks.tau_voice import install

        install()
        from tau2.agent.base.voice import synthesize_voice

        assert synthesize_voice.__name__ == "local_synthesis"
        print("Tau: real voice provider and local synthesis adapter imported", flush=True)
    else:
        from soulx_agent.benchmarks.eva import install

        install()
        from eva.models.config import RunConfig
        from eva.models.record import EvaluationRecord

        config = RunConfig(
            model_list=[
                {
                    "model_name": "caller",
                    "litellm_params": {
                        "model": "openai/local",
                        "api_base": "http://127.0.0.1:8011/v1",
                        "api_key": "local",
                    },
                }
            ],
            framework="openai_realtime",
            model={
                "s2s": "soulx_claude_code",
                "s2s_params": {"api_key": "local", "model": "soulx_claude_code"},
            },
            user_simulator={"provider": "local"},
            metrics=[],
            preflight=False,
        )
        assert config.user_simulator.provider == "local"
        for domain, count in [("airline", 50), ("itsm", 80), ("medical_hr", 83)]:
            records = EvaluationRecord.load_dataset(
                args.dataset_dir / "eva/data" / f"{domain}_dataset.json"
            )
            assert len(records) == count
        print("EVA: real local-caller config and all 213 scenarios validated", flush=True)


if __name__ == "__main__":
    main()
