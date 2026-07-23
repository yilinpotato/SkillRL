#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-coskill-alfworld-fixed-ablation:skillrl-cu128}"
HOST_OUTPUT_ROOT="${HOST_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/docker_alfworld_ablation}"
HOST_CACHE_ROOT="${HOST_CACHE_ROOT:-$PROJECT_ROOT/.docker-cache}"
ENV_FILE="${COSKILL_ENV_FILE:-$PROJECT_ROOT/.env}"
CONTAINER_MODE="skill-level"
if [[ "${1:-}" =~ ^(cloud-check|shell|ablation|skill-level|trace-compression-off)$ ]]; then
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
done

for key in AB_ROOT TRACE_OUTPUT_DIR OUTPUT_DIR MODEL_PATH VLLM_ENFORCE_EAGER \
           VLLM_MAX_NUM_SEQS VLLM_GPU_MEMORY_UTILIZATION MAX_EPISODES \
           BATCH_ROLLOUT_SIZE ABLATION_ROLLOUTS_PER_TYPE EVAL_GROUPS_PER_LEVEL \
           EVAL_GAMES_PER_TYPE CLOUD_BOOTSTRAP_PROBE CHECKPOINT_EVERY_GROUPS LOG_TRAJECTORIES; do
  if [[ -n "${!key:-}" ]]; then
    DOCKER_ENV_ARGS+=(-e "$key")
  fi
done

MODEL_MOUNT_ARGS=()
HOST_MODEL_PATH="${HOST_MODEL_PATH:-${MODEL_SOURCE:-}}"
if [[ -n "$HOST_MODEL_PATH" ]]; then
  if [[ ! -f "$HOST_MODEL_PATH/config.json" ]]; then
    echo "HOST_MODEL_PATH is not a complete model directory: $HOST_MODEL_PATH" >&2
    exit 2
  fi
  MODEL_MOUNT_ARGS=(-v "$HOST_MODEL_PATH:/opt/models/Qwen3-4B-Thinking-2507:ro")
  DOCKER_ENV_ARGS+=(-e MODEL_PATH=/opt/models/Qwen3-4B-Thinking-2507)
fi

EXTERNAL_TRACE_MOUNT_ARGS=()
HOST_EXTERNAL_RAW_TRACES="${HOST_EXTERNAL_RAW_TRACES:-${EXTERNAL_RAW_TRACES:-}}"
if [[ -n "$HOST_EXTERNAL_RAW_TRACES" ]]; then
  if [[ ! -f "$HOST_EXTERNAL_RAW_TRACES" ]]; then
    echo "HOST_EXTERNAL_RAW_TRACES is not a readable JSONL file: $HOST_EXTERNAL_RAW_TRACES" >&2
    exit 2
  fi
  EXTERNAL_TRACE_CONTAINER_PATH="/workspace/external/raw_traces.jsonl"
  EXTERNAL_TRACE_MOUNT_ARGS=(-v "$HOST_EXTERNAL_RAW_TRACES:$EXTERNAL_TRACE_CONTAINER_PATH:ro")
  DOCKER_ENV_ARGS+=(-e "EXTERNAL_RAW_TRACES=$EXTERNAL_TRACE_CONTAINER_PATH")
fi

GPU_ARGS=()
if [[ "$CONTAINER_MODE" != "cloud-check" ]]; then
  GPU_SPEC="${CUDA_VISIBLE_DEVICES:-}"
  GPU1_NAME="$(nvidia-smi --id=1 --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  if [[ "${COSKILL_FORCE_LOCAL_3090:-0}" == "1" || "$GPU1_NAME" == *"RTX 3090"* ]]; then
    [[ -z "$GPU_SPEC" || "$GPU_SPEC" == "1" ]] || {
      echo "Shared 3090 host may only use physical GPU 1." >&2; exit 2;
    }
    active="$(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF' || true)"
    [[ -z "$active" ]] || { echo "Local GPU 1 is in use by PID(s): $active" >&2; exit 2; }
    GPU_SPEC="1"
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
  "${MODEL_MOUNT_ARGS[@]}" \
  "${EXTERNAL_TRACE_MOUNT_ARGS[@]}" \
  "${DOCKER_ENV_ARGS[@]}" \
  "$IMAGE_NAME" "$CONTAINER_MODE" "$@"
