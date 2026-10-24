#!/bin/bash
#SBATCH --job-name=soulx_data
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/soulx-agent/bin/activate"
export PYTHONUNBUFFERED=1
cd "${PROJECT_DIR}/soulx-audio-agent" || exit 1
python -m scripts.prepare_datasets \
    --source_dir "${SCRATCH_DIR}/soulx-audio-agent/sources" \
    --dataset_dir "${SCRATCH_DIR}/datasets"
