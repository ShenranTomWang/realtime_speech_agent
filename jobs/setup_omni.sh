#!/bin/bash
#SBATCH --job-name=soulx_omni_setup
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/cue-gpu/bin/activate"
export PYTHONUNBUFFERED=1
cd "${PROJECT_DIR}/soulx-audio-agent" || exit 1
# Install runtime dependencies without automated test checks.
bash -c "$(cat scripts/setup_omni.sh)"
