#!/bin/bash
#SBATCH --job-name=soulx_frontend
#SBATCH --partition=nlpgpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:nvidia_rtx_a6000:1

source "${HOME_DIR}/.bashrc"
source "${SCRATCH_DIR}/envs/soulx-agent/bin/activate"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "${PROJECT_DIR}/soulx-audio-agent" || exit 1
python -m scripts.host_frontend \
    --gpu_layout shared \
    --llm_config "${SOULX_LLM_CONFIG:-config/llm/qwen35_4b_sglang_vector.yaml}" \
    --frontend_tools_override "${SOULX_FRONTEND_TOOLS_OVERRIDE:-config/frontend_tools/delegate_simple.yaml}" \
    --source_dir "${SCRATCH_DIR}/soulx-audio-agent/sources" \
    --output_dir "${SCRATCH_DIR}/soulx-audio-agent/hosting/${SLURM_JOB_ID}" \
    --cloudflared "${SOULX_CLOUDFLARED:-${SCRATCH_DIR}/soulx-audio-agent/bin/cloudflared}"
