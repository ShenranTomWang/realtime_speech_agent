"""Process ownership shared by the audio stack and standalone LLM runner."""

import contextlib
import os
import signal
import subprocess
from pathlib import Path


def launch(environment, command, cwd):
    label = f"{environment or 'current'}:{command[0]}"
    if environment:
        activation = Path(os.environ["SCRATCH_DIR"]) / "envs" / environment / "bin/activate"
        if not activation.is_file():
            raise FileNotFoundError(f"Environment not prepared: {activation}")
        command = [
            "bash",
            "-c",
            'source "${HOME_DIR}/.bashrc"; source "$1" || exit; shift; exec "$@"',
            "bash",
            str(activation),
            *command,
        ]
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    print(f"Started {label} pid={process.pid}", flush=True)
    return process


def stop_processes(processes):
    for process in reversed(processes):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
