#!/bin/bash
# CUDA 12.8 model executor for the cluster's 570-series driver; Omni has a separate CPU environment.
source "${HOME_DIR}/.bashrc"
if [[ ! -f "${SCRATCH_DIR}/envs/soulx-sglang/bin/activate" ]]; then
    uv venv --python 3.12 "${SCRATCH_DIR}/envs/soulx-sglang" || exit 1
fi
source "${SCRATCH_DIR}/envs/soulx-sglang/bin/activate"
uv pip install 'sglang==0.5.10' 'torch==2.9.1' \
    'torchvision==0.24.1' 'torchaudio==2.9.1' \
    --torch-backend=cu128 --prerelease=allow || exit 1
python -c 'import importlib.metadata as m; print({name: m.version(name) for name in ["sglang", "torch", "torchvision"]})'
