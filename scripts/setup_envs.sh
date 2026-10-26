#!/bin/bash
# Run on an allocated build/compute node, from the project directory.
source "${HOME_DIR}/.bashrc"

prepare_environment() {
    if [[ ! -f "${SCRATCH_DIR}/envs/$1/bin/activate" ]]; then
        uv venv --python "$2" "${SCRATCH_DIR}/envs/$1" || return 1
    fi
    source "${SCRATCH_DIR}/envs/$1/bin/activate"
}

prepare_environment soulx-agent 3.12 || exit 1
uv sync --active --locked || exit 1
uv pip install huggingface_hub modelscope gdown || exit 1
python -m scripts.setup_sources \
    --source_dir "${SCRATCH_DIR}/soulx-audio-agent/sources" \
    --download_models || exit 1

if [[ "${SOULX_INSTALL_VLLM:-0}" == "1" ]]; then
    prepare_environment soulx-llm 3.12 || exit 1
    uv pip install vllm --torch-backend=auto || exit 1
fi

prepare_environment soulx-gateway 3.12 || exit 1
uv sync --active --locked --extra gateway || exit 1

prepare_environment soulx-vad 3.10 || exit 1
uv pip install -r config/vad_requirements.txt || exit 1

prepare_environment soulx-tts 3.12 || exit 1
uv pip install \
    -r "${SCRATCH_DIR}/soulx-audio-agent/sources/SoulX-Duplug/dialogue_system/modules/index_tts_vllm/requirements.txt" || exit 1

benchmarks=(
    tau
    eva
    fdb
)
for benchmark in "${benchmarks[@]}"; do
    prepare_environment "soulx-${benchmark}" 3.12 || exit 1
    case "$benchmark" in
        tau)
            if [[ ! -f "${VIRTUAL_ENV}/include/portaudio.h" ]]; then
                (
                    cd "${SCRATCH_DIR}/soulx-audio-agent/sources/portaudio" || exit 1
                    ./configure --prefix="${VIRTUAL_ENV}" \
                        --without-alsa --without-jack --without-oss \
                        --disable-shared CFLAGS=-fPIC || exit 1
                    make -j8 || exit 1
                    make install || exit 1
                ) || exit 1
            fi
            CFLAGS="-I${VIRTUAL_ENV}/include" LDFLAGS="-L${VIRTUAL_ENV}/lib" uv pip install -e . \
                -e "${SCRATCH_DIR}/soulx-audio-agent/sources/tau2-bench[voice]" || exit 1
            ;;
        eva)
            uv pip install -e . \
                -e "${SCRATCH_DIR}/soulx-audio-agent/sources/eva" || exit 1
            ;;
        fdb)
            uv pip install -e . \
                openai python-dotenv pydub ffmpeg-python || exit 1
            if [[ "${SOULX_INSTALL_EVAL:-0}" == "1" ]]; then
                uv pip install 'nemo_toolkit[asr]' || exit 1
            fi
            ;;
    esac
done

prepare_environment soulx-agent 3.12 || exit 1
# Stage release-compatible English metadata alongside the original HF release.
python -m scripts.prepare_datasets \
    --source_dir "${SCRATCH_DIR}/soulx-audio-agent/sources" \
    --dataset_dir "${SCRATCH_DIR}/datasets" || exit 1
