#!/usr/bin/env bash
set -euo pipefail

# Bounded DGX Spark (NVIDIA GB10) vLLM launcher for Cronus-9B.
# Download weights first:
#   hf download Sn1waR/Cronus-9B --local-dir /path/to/Cronus-9B

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${CRONUS_BUNDLE_DIR:-${1:-}}"
IMAGE="${CRONUS_VLLM_IMAGE:-cronus-vllm:latest}"
CONTAINER_NAME="${CRONUS_CONTAINER_NAME:-muteki-guard-cronus}"
PORT="${CRONUS_PORT:-18971}"

if [[ -z "${BUNDLE_DIR}" ]]; then
  echo "Usage: CRONUS_BUNDLE_DIR=/path/to/Cronus-9B $0" >&2
  exit 2
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --user 1000:1000 \
  --memory=48g \
  --memory-swap=48g \
  --pids-limit=1024 \
  --shm-size=8g \
  -p "${PORT}:${PORT}" \
  -e CUDA_MEM_FRACTION=0.33 \
  -e GUARD_GPU_MEM_UTIL=0.30 \
  -e CRONUS_BUNDLE_DIR=/models/Cronus-9B \
  -e GUARD_PORT="${PORT}" \
  -e PYTHONPATH=/workspace/src \
  -v "${REPO_ROOT}:/workspace:ro" \
  -v "${BUNDLE_DIR}:/models/Cronus-9B:ro" \
  "${IMAGE}" \
  python3 -m cronus.deploy.serve_vllm
