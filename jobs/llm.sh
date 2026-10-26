#!/bin/bash
#SBATCH --job-name=soulx_llm
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:nvidia_rtx_a6000:1

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/soulx-agent/bin/activate"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SOULX_STARTUP_TIMEOUT="${SOULX_STARTUP_TIMEOUT:-1800}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_DIR}/soulx-audio-agent" || exit 1

# Start runtime services with readiness waits; debug through interactive sessions.
python -m scripts.run_llm \
    --llm_config "${SOULX_LLM_CONFIG:-config/llm/managed_local.yaml}" \
    --output_dir "${SOULX_LLM_OUTPUT_DIR:-${SCRATCH_DIR}/soulx-audio-agent/llm_runtime}"
