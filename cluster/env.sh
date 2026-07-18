#!/usr/bin/env bash
# Shared paths. Values supplied by the submitter take precedence.
export PROJECT_ROOT="${PROJECT_ROOT:-/hpc/group/honglab/zb78/ft-whisper/mandarin-lyrics-whisper}"
export STORAGE_ROOT="${STORAGE_ROOT:-/hpc/group/honglab/zb78}"
export DATA_ROOT="${DATA_ROOT:-${STORAGE_ROOT}/dataset/whisper-song}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/model/whisper-song}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${STORAGE_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STORAGE_ROOT}/cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${STORAGE_ROOT}/cache/whisper-song/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${STORAGE_ROOT}/cache/conda}"
export VIRTUAL_ENV_PATH="${VIRTUAL_ENV_PATH:-${STORAGE_ROOT}/env/whisper-song}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

[[ -d "$PROJECT_ROOT" ]] || { echo "PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/gates" "$HF_HOME" "$DATA_ROOT" "$PIP_CACHE_DIR"
