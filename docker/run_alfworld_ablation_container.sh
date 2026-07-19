#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-coskill-alfworld-fixed-ablation:skillrl-cu128}"
HOST_OUTPUT_ROOT="${HOST_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/docker_alfworld_ablation}"
HOST_CACHE_ROOT="${HOST_CACHE_ROOT:-$PROJECT_ROOT/.docker-cache}"
ENV_FILE="${COSKILL_ENV_FILE:-$PROJECT_ROOT/.env}"
CONTAINER_MODE="ablation"
if [[ "${1:-}" == "cloud-check" || "${1:-}" == "shell" ]]; then
  CONTAINER_MODE="$1"
  shift
fi

mkdir -p "$HOST_OUTPUT_ROOT" "$HOST_CACHE_ROOT"
DOCKER_ENV_ARGS=()
if [[ -f "$ENV_FILE" ]]; then
  PRIVATE_ENV_FILE="$ENV_FILE"
  source "$PROJECT_ROOT/scripts/load_private_env.sh"
  unset PRIVATE_ENV_FILE
fi
for key in DEEPSEEK_API_KEY DEEPSEEK_MODEL DEEPSEEK_API_BASE SKILL_UPDATER_BACKEND \
           AZURE_OPENAI_API_KEY AZURE_OPENAI_ENDPOINT AZURE_OPENAI_API_VERSION AZURE_OPENAI_MODEL; do
  if [[ -n "${!key:-}" ]]; then
    # ``-e KEY`` copies the already sourced value without placing the secret
    # itself in this script's command line or image history.
    DOCKER_ENV_ARGS+=(-e "$key")
  fi
fi

GPU_ARGS=()
if [[ "$CONTAINER_MODE" != "cloud-check" ]]; then
  GPU_SPEC="${CUDA_VISIBLE_DEVICES:-}"
  GPU0_NAME="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  if [[ "${COSKILL_FORCE_LOCAL_3090:-0}" == "1" || "$GPU0_NAME" == *"RTX 3090"* ]]; then
    [[ -z "$GPU_SPEC" || "$GPU_SPEC" == "0" ]] || {
      echo "Shared 3090 host may only use physical GPU 0." >&2; exit 2;
    }
    active="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF' || true)"
    [[ -z "$active" ]] || { echo "Local GPU 0 is in use by PID(s): $active" >&2; exit 2; }
    GPU_SPEC="0"
  elif [[ -z "$GPU_SPEC" ]]; then
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF{n++} END{print n+0}')"
    (( GPU_COUNT > 0 )) || { echo "No NVIDIA GPU detected." >&2; exit 2; }
    (( GPU_COUNT > 8 )) && GPU_COUNT=8
    GPU_SPEC="$(seq -s, 0 $((GPU_COUNT - 1)))"
  fi
  GPU_ARGS=(--gpus "\"device=$GPU_SPEC\"")
fi

docker run --rm --ipc=host --shm-size=16g \
  "${GPU_ARGS[@]}" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOST_OUTPUT_ROOT:/workspace/outputs" \
  -v "$HOST_CACHE_ROOT:/workspace/cache" \
  "${DOCKER_ENV_ARGS[@]}" \
  "$IMAGE_NAME" "$CONTAINER_MODE" "$@"
