#!/bin/bash
#SBATCH --job-name=soulx_bench_check
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/soulx-tau/bin/activate"
cd "${PROJECT_DIR}/soulx-audio-agent" || exit 1
python -m scripts.check_benchmark_setup --benchmark tau_voice \
    --dataset_dir "${SCRATCH_DIR}/datasets" || exit 1
source "${SCRATCH_DIR}/envs/soulx-eva/bin/activate"
python -m scripts.check_benchmark_setup --benchmark eva \
    --dataset_dir "${SCRATCH_DIR}/datasets" || exit 1
