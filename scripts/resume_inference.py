"""Submit benchmark/profile recovery chains without waiting for their jobs."""

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--submission_record", type=Path, required=True)
    args = parser.parse_args()
    profiles = {
        "all_tools": "",
        "delegate_only": "config/frontend_tools/delegate_only.yaml",
        "delegate_simple": "config/frontend_tools/delegate_simple.yaml",
    }
    record = {
        "output_root": str(args.output_root),
        "profiles": {},
        "dependency_policy": "afterok per benchmark",
    }
    predecessors = {}
    args.submission_record.parent.mkdir(parents=True, exist_ok=True)
    for profile, override in profiles.items():
        record["profiles"][profile] = {}
        for benchmark in ("tau_voice", "eva", "fdb_v3"):
            environment = dict(os.environ)
            environment["SOULX_OUTPUT_ROOT"] = str(args.output_root / profile)
            environment["SOULX_FRONTEND_TOOLS_OVERRIDE"] = override
            # Use the benchmark's fixed port namespace; profiles run sequentially.
            environment.pop("SOULX_PORT_OFFSET", None)
            command = ["/opt/slurm/bin/sbatch", "--parsable"]
            if benchmark in predecessors:
                command.append("--dependency=afterok:" + predecessors[benchmark])
            command.append(benchmark + ".sh")
            result = subprocess.run(
                command,
                cwd=ROOT / "jobs/inference",
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdecimal():
                raise RuntimeError("Slurm returned an invalid job ID")
            record["profiles"][profile][benchmark] = job_id
            predecessors[benchmark] = job_id
            args.submission_record.write_text(json.dumps(record, indent=2) + "\n")
            print(f"{profile}/{benchmark}: submitted {job_id}", flush=True)


if __name__ == "__main__":
    main()
