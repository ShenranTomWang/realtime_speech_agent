#!/bin/bash
#SBATCH --job-name=soulx_fdb_v3_delegate_simple_eval
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:nvidia_rtx_a6000:1

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/soulx-fdb/bin/activate"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONUNBUFFERED=1
export SOULX_FDB_TAIL_SECONDS=300
export SOULX_FRONTEND_RUNTIME="${SOULX_FRONTEND_RUNTIME:-sglang_omni}"
export SOULX_OMNI_CONFIG="${PROJECT_DIR}/soulx-audio-agent/config/omni.yaml"
export TOKENIZERS_PARALLELISM=false
export SOULX_PORT_OFFSET=2200
export SOULX_FRONTEND_TOOLS_OVERRIDE="config/frontend_tools/delegate_simple.yaml"
export SOULX_OUTPUT_ROOT="${SOULX_OUTPUT_ROOT:-${SCRATCH_DIR}/soulx-audio-agent/results/delegate_simple}"
cd "${PROJECT_DIR}/soulx-audio-agent" || exit 1

# Preserve recorded failures and continue across benchmark domains.
python -m scripts.run_benchmark fdb_v3 eval \
    --caller_mode "${SOULX_CALLER_MODE:-local}" \
    --llm_config "${SOULX_LLM_CONFIG:-config/llm/qwen35_4b_sglang_vector.yaml}" \
    --source_dir "${SCRATCH_DIR}/soulx-audio-agent/sources" \
    --output_dir "${SOULX_OUTPUT_ROOT:-${SCRATCH_DIR}/soulx-audio-agent/results}/fdb_v3" \
    --data_dir "${SCRATCH_DIR}/datasets/fdb_v3/fdb_v3_data_released"
