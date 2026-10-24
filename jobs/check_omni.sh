#!/bin/bash
#SBATCH --job-name=soulx_omni_check
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/soulx-omni/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
cd "${PROJECT_DIR}/soulx-audio-agent" || exit 1
python -c 'from soulx_agent.omni_runtime import OmniRuntime; OmniRuntime("config/omni.yaml"); print("Production Omni graph validated")' || exit 1
python -m pytest tests/test_omni_runtime.py -vv -s || exit 1
