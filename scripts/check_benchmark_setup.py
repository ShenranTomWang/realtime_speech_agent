"""Validate actual upstream schemas and local-caller adapters without running models."""

import argparse
import json
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
        from tau2.metrics.benchmark import _authoritative_language
        from tau2.multilingual.runtime import reload_call_language, resolve_task
        from tau2.runner.presets import binding_revision, load_cell_language

        for domain in ("airline", "retail", "telecom"):
            loaded = load_cell_language("en", domain, "voice", canonical_tasks=True)
            context = resolve_task(loaded, loaded.tasks[0], seed=0)
            restored = reload_call_language(context, domain)
            assert restored.artifact_pins == loaded.artifact_pins
            _authoritative_language("en", domain, binding_revision(loaded), "voice")
            changed = context.model_copy(update={"tasks_sha256": "0" * 64})
            try:
                reload_call_language(changed, domain)
            except ValueError:
                pass
            else:
                raise AssertionError("Tau accepted modified task artifact pins")
            print(f"Tau {domain}: canonical reload/export and drift rejection passed", flush=True)
    else:
        from soulx_agent.benchmarks.eva import install

        install()
        os.environ["SOULX_EVA_DATA_DIR"] = str((args.dataset_dir / "eva").resolve())
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
            from scripts.prepare_datasets import eva_record

            release_name = {"airline": "csm_airline", "itsm": "itsm", "medical_hr": "hr_medical"}[
                domain
            ]
            release_path = args.dataset_dir / "eva/release/data" / f"eva_bench_{release_name}.jsonl"
            expected = [
                eva_record(json.loads(line))
                for line in release_path.read_text().splitlines()
                if line.strip()
            ]
            staged = json.loads(
                (args.dataset_dir / "eva/data" / f"{domain}_dataset.json").read_text()
            )
            assert staged == expected, f"EVA {domain} differs from the pinned English release"
            from eva.utils.culture import resolve_scenario_db, resolve_user_goal

            for record in records:
                goal = resolve_user_goal(
                    record.user_goal,
                    record.culture_overrides,
                    "en",
                    record.romanized_culture_overrides,
                    record.starting_utterances,
                )
                assert goal["starting_utterance"] == record.user_goal["starting_utterance"]
                path = args.dataset_dir / "eva/data" / f"{domain}_scenarios" / f"{record.id}.json"
                db = json.loads(path.read_text())
                resolved = resolve_scenario_db(
                    db,
                    record.culture_overrides,
                    "en",
                    record.romanized_culture_overrides,
                )
                assert resolved == db
            print(f"EVA {domain}: all {count} literal English goals/databases resolved", flush=True)
        print("EVA: real local-caller config and all 213 scenarios validated", flush=True)


if __name__ == "__main__":
    main()
