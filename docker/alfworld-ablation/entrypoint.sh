#!/usr/bin/env bash
set -euo pipefail

export COSKILL_CONTAINER=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

COSKILL_PROJECT_ROOT="${COSKILL_PROJECT_ROOT:-/workspace/CoSkill}"
COSKILL_CACHE_ROOT="${COSKILL_CACHE_ROOT:-/workspace/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs}"
ALFWORLD_DATA="${ALFWORLD_DATA:-/opt/data/alfworld}"
BAKED_MODEL_PATH="${BAKED_MODEL_PATH:-/opt/models/Qwen3-4B-Thinking-2507}"
MODELSCOPE_MODEL_ID="${MODELSCOPE_MODEL_ID:-Qwen/Qwen3-4B-Thinking-2507}"

if [[ "${1:-}" == "cloud-check" ]]; then
  shift
  cd "$COSKILL_PROJECT_ROOT"
  exec python scripts/check_cloud_bootstrap.py \
    --environment alfworld \
    --skills-json memory_data/alfworld/claude_style_skills.json \
    "$@"
fi

mkdir -p "$COSKILL_CACHE_ROOT" "$OUTPUT_ROOT"
if [[ -f "$BAKED_MODEL_PATH/config.json" ]]; then
  MODEL_PATH="${MODEL_PATH:-$BAKED_MODEL_PATH}"
else
  MODEL_PATH="${MODEL_PATH:-$COSKILL_CACHE_ROOT/models/Qwen3-4B-Thinking-2507}"
fi
export COSKILL_PROJECT_ROOT COSKILL_CACHE_ROOT OUTPUT_ROOT ALFWORLD_DATA MODEL_PATH MODELSCOPE_MODEL_ID

if ! find "$ALFWORLD_DATA/json_2.1.1/train" -name traj_data.json -print -quit 2>/dev/null | grep -q .; then
  echo "ALFWorld data not found under $ALFWORLD_DATA." >&2
  echo "Build with INCLUDE_ALFWORLD=1 or mount it with -v /path/to/alfworld:$ALFWORLD_DATA:ro." >&2
  exit 2
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  if [[ "${COSKILL_AUTO_DOWNLOAD_MODEL:-1}" != "1" ]]; then
    echo "Model missing at $MODEL_PATH and automatic ModelScope download is disabled." >&2
    exit 2
  fi
  export HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0
  python "$COSKILL_PROJECT_ROOT/docker/alfworld-ablation/download_model.py"
  export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
fi

detect_gpu_count() {
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "$visible" && "$visible" != "all" && "$visible" != "void" ]]; then
    awk -F',' '{print NF}' <<<"$visible"
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF{n++} END{print n+0}'
}

GPU_COUNT="$(detect_gpu_count)"
MAX_GPUS="${COSKILL_MAX_GPUS:-8}"
if (( GPU_COUNT < 1 )); then
  echo "No CUDA GPU is visible inside the container. Start it with Docker --gpus." >&2
  exit 2
fi
if (( GPU_COUNT > MAX_GPUS )); then
  GPU_COUNT="$MAX_GPUS"
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES:-}" == "all" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
fi
export CUDA_VISIBLE_DEVICES

# One full 4B vLLM replica per visible GPU is fastest for these independent
# rollouts. TP/PP remain 1; total evaluation rollout count remains exactly 36.
DATA_PARALLEL_WORKERS="${DATA_PARALLEL_WORKERS:-$GPU_COUNT}"
ROLLOUT_WORKER_GPUS="${ROLLOUT_WORKER_GPUS:-$CUDA_VISIBLE_DEVICES}"
AB_ROOT="${AB_ROOT:-$OUTPUT_ROOT/alfworld_fixed_trajectory_ablation}"
export DATA_PARALLEL_WORKERS ROLLOUT_WORKER_GPUS AB_ROOT

echo "CoSkill container: GPUs=$CUDA_VISIBLE_DEVICES DP=$DATA_PARALLEL_WORKERS TP=1"
echo "ALFWORLD_DATA=$ALFWORLD_DATA"
echo "MODEL_PATH=$MODEL_PATH"
echo "AB_ROOT=$AB_ROOT"

cd "$COSKILL_PROJECT_ROOT"
case "${1:-ablation}" in
  ablation)
    shift || true
    exec python -u -m examples.playbook_evolve.fixed_trajectory_ablation \
      --root "$AB_ROOT" \
      --alfworld_data "$ALFWORLD_DATA" \
      --model_path "$MODEL_PATH" \
      --data_parallel_workers "$DATA_PARALLEL_WORKERS" \
      --rollout_worker_gpus "$ROLLOUT_WORKER_GPUS" \
      "$@"
    ;;
  shell)
    exec bash
    ;;
  *)
    exec "$@"
    ;;
esac
