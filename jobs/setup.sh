#!/bin/bash
#SBATCH --job-name=soulx_setup
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/cue-gpu/bin/activate"
export PYTHONUNBUFFERED=1
cd "${PROJECT_DIR}/soulx-audio-agent" || exit 1
# Install runtime and FDB scoring dependencies without automated test checks.
bash -c "$(cat scripts/setup_envs.sh)"
