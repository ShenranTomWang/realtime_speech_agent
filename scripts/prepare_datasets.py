"""Stage pinned benchmark releases under scratch and normalize FDB audio layout."""

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

EVA_REVISION = "ab189dd1d0e834d205239e1493c089d0e95f7dda"
FDB_URL = "https://drive.google.com/file/d/1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz/view"


def copy_release(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def prepare(source_dir, dataset_dir):
    copy_release(source_dir / "tau2-bench/data", dataset_dir / "tau_voice")
    eva = dataset_dir / "eva"
    copy_release(source_dir / "eva/data", eva / "data")
    copy_release(source_dir / "eva/configs", eva / "configs")
    from huggingface_hub import snapshot_download

    snapshot_download(
        "ServiceNow-AI/eva-bench",
        repo_type="dataset",
        revision=EVA_REVISION,
        local_dir=eva / "release",
    )
    counts = {}
    for domain, release_name in {
        "airline": "csm_airline",
        "itsm": "itsm",
        "medical_hr": "hr_medical",
    }.items():
        release_file = eva / "release/data" / f"eva_bench_{release_name}.jsonl"
        rows = [json.loads(line) for line in release_file.read_text().splitlines() if line.strip()]
        for row in rows:
            record_id = row["id"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", record_id):
                raise ValueError("Invalid EVA record ID")
            initial_db = row["initial_scenario_database"]
            if not isinstance(initial_db, dict):
                raise ValueError("EVA initial scenario database must be an object")
            scenario = eva / "data" / f"{domain}_scenarios" / f"{record_id}.json"
            scenario.parent.mkdir(parents=True, exist_ok=True)
            scenario.write_text(json.dumps(initial_db, indent=2) + "\n")
        (eva / "data" / f"{domain}_dataset.json").write_text(json.dumps(rows, indent=2) + "\n")
        counts[domain] = len(rows)
    (eva / "manifest.json").write_text(
        json.dumps(
            {"repo": "ServiceNow-AI/eva-bench", "revision": EVA_REVISION, "counts": counts},
            indent=2,
        )
        + "\n"
    )
    print(f"Hugging Face EVA-Bench staged: {counts}", flush=True)
    fdb = dataset_dir / "fdb_v3"
    fdb.mkdir(parents=True, exist_ok=True)
    archive = fdb / "fdb_v3_data_released.zip"
    if not archive.is_file():
        import gdown

        partial = archive.with_suffix(".zip.partial")
        result = gdown.download(
            id="1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz", output=str(partial), resume=True
        )
        if not result:
            raise RuntimeError("Official FDB download failed")
        partial.rename(archive)
    destination = fdb / "fdb_v3_data_released"
    with zipfile.ZipFile(archive) as release:
        for member in release.infolist():
            path = Path(member.filename)
            if path.name not in {"input.wav", "metadata.json"} or "__MACOSX" in path.parts:
                continue
            recording = path.parent.name
            if not re.fullmatch(r".+_[0-9a-f]{24}", recording):
                raise ValueError(f"Unexpected FDB recording directory: {recording}")
            target = destination / recording / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != member.file_size:
                with release.open(member) as incoming, target.open("wb") as output:
                    shutil.copyfileobj(incoming, output)
    recordings = sorted(destination.glob("*/input.wav"))
    if not recordings:
        raise RuntimeError("Official FDB archive contains no recognized recordings")
    manifest = {
        "source": FDB_URL,
        "recordings": [
            {
                "path": str(path.relative_to(destination)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in recordings
        ],
    }
    (fdb / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Benchmark data staged in {dataset_dir}; FDB recordings: {len(recordings)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_dir.resolve(), args.dataset_dir.resolve())


if __name__ == "__main__":
    main()
