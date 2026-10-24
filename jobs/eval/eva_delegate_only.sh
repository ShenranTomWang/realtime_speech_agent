#!/bin/bash
#SBATCH --job-name=soulx_eva_delegate_only_eval
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/soulx-eva/bin/activate"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONUNBUFFERED=1
export SOULX_FRONTEND_RUNTIME="${SOULX_FRONTEND_RUNTIME:-sglang_omni}"
export SOULX_OMNI_CONFIG="${PROJECT_DIR}/soulx-audio-agent/config/omni.yaml"
export TOKENIZERS_PARALLELISM=false
export SOULX_PORT_OFFSET=1100
export SOULX_FRONTEND_TOOLS_OVERRIDE="config/frontend_tools/delegate_only.yaml"
export SOULX_OUTPUT_ROOT="${SCRATCH_DIR}/soulx-audio-agent/results/delegate_only"
cd "${PROJECT_DIR}/soulx-audio-agent" || exit 1

python -m scripts.run_benchmark eva eval \
    --caller_mode "${SOULX_CALLER_MODE:-local}" \
    --llm_config "${SOULX_LLM_CONFIG:-config/llm/qwen35_4b_sglang_vector.yaml}" \
    --source_dir "${SCRATCH_DIR}/soulx-audio-agent/sources" \
    --output_dir "${SOULX_OUTPUT_ROOT:-${SCRATCH_DIR}/soulx-audio-agent/results}/eva" \
    --data_dir "${SCRATCH_DIR}/datasets/eva"
