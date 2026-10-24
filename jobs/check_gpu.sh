#!/bin/bash
#SBATCH --job-name=soulx_gpu_check
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:02:00
#SBATCH --gres=gpu:nvidia_rtx_a6000:1

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/cue-gpu/bin/activate"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
