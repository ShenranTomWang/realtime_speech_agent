#!/bin/bash
#SBATCH --job-name=soulx_sglang_setup
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/cue-gpu/bin/activate"
export PYTHONUNBUFFERED=1
cd "${PROJECT_DIR}/soulx-audio-agent" || exit 1
bash -c "$(cat scripts/setup_sglang.sh)"
