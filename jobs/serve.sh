#!/bin/bash
#SBATCH --job-name=soulx_voice
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:nvidia_rtx_a6000:1

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/soulx-agent/bin/activate"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SOULX_STARTUP_TIMEOUT="${SOULX_STARTUP_TIMEOUT:-1800}"
cd "${PROJECT_DIR}/soulx-audio-agent" || exit 1
python -m scripts.run_stack --benchmark serve \
    --caller_mode "${SOULX_CALLER_MODE:-local}" \
    --frontend_runtime "${SOULX_FRONTEND_RUNTIME:-sglang_omni}" \
    --llm_config "${SOULX_LLM_CONFIG:-config/llm/qwen35_4b_sglang_vector.yaml}" \
    --source_dir "${SCRATCH_DIR}/soulx-audio-agent/sources" \
    --output_dir "${SCRATCH_DIR}/soulx-audio-agent/interactive"
