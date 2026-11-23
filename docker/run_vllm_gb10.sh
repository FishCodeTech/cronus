#!/usr/bin/env bash
set -euo pipefail

# Bounded DGX Spark (NVIDIA GB10) vLLM launcher for Cronus-9B.
# Download weights first:
#   hf download Sn1waR/Cronus-9B --local-dir /path/to/Cronus-9B

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${CRONUS_BUNDLE_DIR:-${1:-}}"
IMAGE="${CRONUS_VLLM_IMAGE:-cronus-vllm:latest}"
CONTAINER_NAME="${CRONUS_CONTAINER_NAME:-muteki-guard-cronus}"
HOST="${CRONUS_HOST:-0.0.0.0}"
PORT="${CRONUS_PORT:-18971}"
ALLOW_CACHE_RESET="${CRONUS_ALLOW_CACHE_RESET:-0}"
API_KEY="${CRONUS_API_KEY:-}"
API_KEY_FILE="${CRONUS_API_KEY_FILE:-}"
CACHE_ROOT="${CRONUS_CACHE_ROOT:-}"

if [[ -z "${BUNDLE_DIR}" ]]; then
  echo "Usage: CRONUS_BUNDLE_DIR=/path/to/Cronus-9B $0" >&2
  exit 2
fi

DOCKER_ARGS=(
  --name "${CONTAINER_NAME}"
  --gpus all
  --user 1000:1000
  --memory=48g
  --memory-swap=48g
  --pids-limit=1024
  --shm-size=8g
  -p "${PORT}:${PORT}"
  -e CUDA_MEM_FRACTION=0.33
  -e GUARD_GPU_MEM_UTIL=0.30
  -e CRONUS_BUNDLE_DIR=/models/Cronus-9B
  -e CRONUS_HOST="${HOST}"
  -e CRONUS_PORT="${PORT}"
  -e GUARD_PORT="${PORT}"
  -e CRONUS_ALLOW_CACHE_RESET="${ALLOW_CACHE_RESET}"
  -e PYTHONPATH=/workspace/src
  -v "${REPO_ROOT}:/workspace:ro"
  -v "${BUNDLE_DIR}:/models/Cronus-9B:ro"
)

if [[ -n "${API_KEY}" ]]; then
  DOCKER_ARGS+=(-e "CRONUS_API_KEY=${API_KEY}")
fi

if [[ -n "${API_KEY_FILE}" ]]; then
  DOCKER_ARGS+=(
    -v "${API_KEY_FILE}:/run/secrets/cronus-api-key:ro"
    -e "CRONUS_API_KEY_FILE=/run/secrets/cronus-api-key"
  )
fi

if [[ -n "${CACHE_ROOT}" ]]; then
  DOCKER_ARGS+=(
    -v "${CACHE_ROOT}/.vllm030-cache/inductor:/tmp/cronus-cache/inductor"
    -v "${CACHE_ROOT}/.vllm030-cache/vllm:/tmp/cronus-cache/vllm"
    -v "${CACHE_ROOT}/.hf-cache:/tmp/cronus-cache/hf"
    -e "TORCHINDUCTOR_CACHE_DIR=/tmp/cronus-cache/inductor"
    -e "VLLM_CACHE_ROOT=/tmp/cronus-cache/vllm"
    -e "HF_HOME=/tmp/cronus-cache/hf"
  )
fi

docker run -d "${DOCKER_ARGS[@]}" "${IMAGE}" \
  python3 -u -m cronus.deploy.serve_vllm
