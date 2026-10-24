#!/bin/bash
# External-service profile: Omni graph/schedulers run on CPU; GPU models have separate environments.
source "${HOME_DIR}/.bashrc"
if [[ ! -f "${SCRATCH_DIR}/envs/soulx-omni/bin/activate" ]]; then
    uv venv --python 3.12 "${SCRATCH_DIR}/envs/soulx-omni" || exit 1
fi
source "${SCRATCH_DIR}/envs/soulx-omni/bin/activate"
uv pip install -e . pyzmq msgpack msgspec xxhash psutil pillow safetensors \
    nvidia-ml-py pytest pytest-asyncio || exit 1
uv pip install 'sglang==0.5.22' --torch-backend=cpu --prerelease=allow || exit 1
# External-service sessions do not decode video. Omit the optional CUDA-only codec.
uv pip uninstall torchcodec torchaudio torchvision || exit 1
uv pip install 'torch==2.14.1' 'torchvision==0.29.1' \
    --torch-backend=cpu || exit 1
python -m scripts.setup_sources \
    --source_dir "${SCRATCH_DIR}/soulx-audio-agent/sources" || exit 1
uv pip install --no-deps \
    -e "${SCRATCH_DIR}/soulx-audio-agent/sources/sglang-omni" || exit 1
python -c 'from soulx_agent.omni_runtime import OmniRuntime; OmniRuntime("config/omni.yaml"); print("Omni graph imports and configuration validated")' || exit 1

python -m pytest tests/test_omni_runtime.py -q || exit 1
