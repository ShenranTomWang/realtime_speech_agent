"""Underscore-style CLI wrapping pinned upstream entry points."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from soulx_agent.llm_runner import DEFAULT_CONFIG, load_llm_plan
from soulx_agent.provenance import verify_run

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd, environment=None):
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["tau_voice", "eva", "fdb_v3"])
    parser.add_argument("phase", choices=["inference", "eval"])
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument(
        "--llm_config", type=Path, default=os.environ.get("SOULX_LLM_CONFIG", DEFAULT_CONFIG)
    )
    parser.add_argument("--tail_seconds", type=float, default=120)
    parser.add_argument(
        "--latency_profile", choices=["instant", "fast", "normal", "slow"], default="instant"
    )
    parser.add_argument("--caller_mode", choices=["local", "service"], default="local")
    args = parser.parse_args()
    os.environ["SOULX_CALLER_MODE"] = args.caller_mode
    os.environ.setdefault(
        "SOULX_PORT_OFFSET", str({"tau_voice": 0, "eva": 100, "fdb_v3": 200}[args.benchmark])
    )
    args.output_dir = args.output_dir.resolve()
    args.source_dir = args.source_dir.resolve()
    verify_run(args.output_dir, args.source_dir, args.benchmark, args.llm_config)
    env = dict(os.environ)
    env["SOULX_BENCHMARK_PHASE"] = args.phase
    caller_arguments = []
    if args.caller_mode == "local":
        plan = load_llm_plan(args.llm_config)
        caller = plan.runners[plan.roles["frontend"]]
        env["SOULX_CALLER_BASE_URL"] = caller.base_url
        env["SOULX_CALLER_MODEL"] = caller.model
        caller_arguments = [
            "--user-llm",
            "openai/" + caller.model,
            "--user-llm-args",
            json.dumps(
                {
                    "api_base": caller.base_url,
                    "api_key": "local",
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                }
            ),
        ]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "integrations"), str(ROOT), env.get("PYTHONPATH", "")]
    )
    if args.benchmark == "tau_voice":
        source = args.source_dir / "tau2-bench"
        if args.data_dir:
            env["TAU2_DATA_DIR"] = str(args.data_dir.resolve())
            os.environ["TAU2_DATA_DIR"] = env["TAU2_DATA_DIR"]
        if args.phase == "inference":
            env["SOULX_BENCHMARK"] = "tau_voice"
            env["LAUNA_API_KEY"] = "local"
            env["LAUNA_WS_URL"] = os.environ.get(
                "SOULX_SERVER_URL", "ws://127.0.0.1:8765/v1/realtime"
            )
            run(
                [
                    sys.executable,
                    "-m",
                    "tau2.cli",
                    "run",
                    "--benchmark",
                    "voice-v1",
                    "--audio-native",
                    "--audio-native-provider",
                    "launa",
                    "--audio-native-model",
                    "soulx_claude_code",
                    "--max-concurrency",
                    "1",
                    "--global-limit",
                    "1",
                    "--workers",
                    "1",
                    *caller_arguments,
                    "--save-to",
                    str(args.output_dir),
                ],
                source,
                env,
            )
        else:
            from tau2.data_model.simulation import Results
            from tau2.metrics.agent_metrics import compute_metrics
            from tau2.metrics.benchmark import load_saved_benchmark
            from tau2.scripts.evaluate_trajectories import compute_simulation_rewards

            reports = {}
            for file in sorted(args.output_dir.rglob("results.json")):
                results = Results.load(file)
                if results.simulations:
                    results = compute_simulation_rewards(results)
                    results.save(file)
                    reports[str(file.relative_to(args.output_dir))] = compute_metrics(
                        results
                    ).model_dump()
            if not reports:
                raise FileNotFoundError("No Tau voice results; run inference first")
            (args.output_dir / "metrics.json").write_text(json.dumps(reports, indent=2))
            summary = load_saved_benchmark(args.output_dir).submission
            (args.output_dir / "benchmark_summary.json").write_text(
                summary.model_dump_json(indent=2)
            )
    elif args.benchmark == "eva":
        source = args.data_dir.resolve() if args.data_dir else args.source_dir / "eva"
        env["SOULX_BENCHMARK"] = "eva"
        env.update(
            {
                "EVA_FRAMEWORK": "openai_realtime",
                "EVA_MODEL__LLM": "",
                "EVA_MODEL__STT": "",
                "EVA_MODEL__TTS": "",
                "EVA_MODEL__S2S": "soulx_claude_code",
                "EVA_MODEL__S2S_PARAMS": '{"api_key":"local","model":"soulx_claude_code"}',
                "EVA_MAX_CONCURRENT_CONVERSATIONS": "1",
                "EVA_OUTPUT_DIR": str(args.output_dir),
            }
        )
        if args.caller_mode == "local":
            env["EVA_USER_SIMULATOR__PROVIDER"] = "local"
            env.setdefault(
                "EVA_MODEL_LIST",
                json.dumps(
                    [
                        {
                            "model_name": "local_caller",
                            "litellm_params": {
                                "model": "openai/" + caller.model,
                                "api_base": caller.base_url,
                                "api_key": "local",
                            },
                        }
                    ]
                ),
            )
        for domain in ["airline", "itsm", "medical_hr"]:
            env["EVA_DOMAIN"] = domain
            env["EVA_RUN_ID"] = "soulx_claude_code_" + domain
            if args.phase == "inference":
                env["EVA_METRICS"] = ""
                env["EVA_PREFLIGHT"] = "false"
            else:
                if not (args.output_dir / env["EVA_RUN_ID"] / "config.json").exists():
                    raise FileNotFoundError("EVA inference run missing")
                env["EVA_METRICS"] = "all"
                env["EVA_MAX_RERUN_ATTEMPTS"] = "0"
                env["EVA_PREFLIGHT"] = "false"
            run([sys.executable, "-m", "eva.cli"], source, env)
    else:
        from soulx_agent.benchmarks.fdb_v3 import infer, prepare_scoring

        if args.data_dir is None:
            parser.error("--data_dir is required for FDB v3")
        source = args.source_dir / "Full-Duplex-Bench/v3"
        if args.phase == "inference":
            asyncio.run(
                infer(
                    source, args.data_dir, args.output_dir, args.tail_seconds, args.latency_profile
                )
            )
        else:
            prepare_scoring(source, args.data_dir, args.output_dir)
            for script, report in [
                ("evaluate_tool_calls.py", "tool_accuracy.json"),
                ("evaluate_pass_rate.py", "pass_rate.json"),
            ]:
                run(
                    [
                        sys.executable,
                        str(source / script),
                        "--benchmark",
                        str(source / "benchmark_data_v2.json"),
                        "--results-dir",
                        str(args.output_dir),
                        "--provider",
                        "soulx_claude_code",
                        "--output",
                        str(args.output_dir / report),
                        "--use-llm",
                    ],
                    source,
                )
            run(
                [
                    sys.executable,
                    str(source / "analyze_tool_latency.py"),
                    "--results-dir",
                    str(args.output_dir),
                    "--provider",
                    "soulx_claude_code",
                    "--output",
                    str(args.output_dir / "latency.json"),
                ],
                source,
            )


if __name__ == "__main__":
    main()
