#!/usr/bin/env bash
# Reuse a completed V4 L0--L5 run and evaluate only newly added held-out games.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-skillRL}"
fi
set -u

source "$PROJECT_ROOT/scripts/load_private_env.sh"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export CACHE_ROOT="${CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
# This explicit post-.env override lets the parallel wrapper bind one process
# per physical GPU even when the private .env defines CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${V4_EXTENSION_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-1}}"

SOURCE_V4_ROOT="${V4_EXTENSION_SOURCE_ROOT:-${SOURCE_V4_ROOT:-}}"
[[ -n "$SOURCE_V4_ROOT" && -f "$SOURCE_V4_ROOT/run_config.json" ]] || {
  echo "Set SOURCE_V4_ROOT to the completed V4 root containing run_config.json." >&2
  exit 2
}
[[ -d "$ALFWORLD_DATA/json_2.1.1" && -f "$MODEL_PATH/config.json" ]] || {
  echo "Missing ALFWORLD_DATA or MODEL_PATH assets." >&2
  exit 2
}

GPU_COUNT="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF{n++} END{print n+0}')"
DP_WORKERS="${DATA_PARALLEL_WORKERS:-$GPU_COUNT}"
[[ "$GPU_COUNT" -ge 1 && "$GPU_COUNT" -le 2 ]] || {
  echo "V4 validation extension supports one or two visible GPUs; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
}
[[ "$DP_WORKERS" == "$GPU_COUNT" ]] || {
  echo "DATA_PARALLEL_WORKERS must equal the visible GPU count." >&2
  exit 2
}
TORCH_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
[[ "$TORCH_GPU_COUNT" == "$GPU_COUNT" ]] || {
  echo "CUDA mask resolves to $TORCH_GPU_COUNT runtime device(s), expected $GPU_COUNT from CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES." >&2
  echo "Inside a one-GPU container, the visible device is normally logical CUDA_VISIBLE_DEVICES=0." >&2
  exit 2
}

VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
AB_ROOT="${V4_EXTENSION_ROOT:-${AB_ROOT:-$PROJECT_ROOT/skillrl_outputs/alfworld_skill_tree_depth_v4_validation_extension/$(date +%Y%m%d_%H%M%S)}}"
if [[ "$(realpath -m "$AB_ROOT")" == "$(realpath "$SOURCE_V4_ROOT")" ]]; then
  echo "AB_ROOT must differ from SOURCE_V4_ROOT; the completed source run is immutable." >&2
  exit 2
fi

echo "[skill-tree-v4-extension] source=$SOURCE_V4_ROOT"
echo "[skill-tree-v4-extension] target=$AB_ROOT"
echo "[skill-tree-v4-extension] GPUs=$CUDA_VISIBLE_DEVICES DP=$DP_WORKERS"
echo "[skill-tree-v4-extension] reuse frozen L0-L5 artifacts; cloud/tree generation disabled"
echo "[skill-tree-v4-extension] requested final games/task=${V4_EVAL_GAMES_PER_TYPE:-5} rollouts/game=${V4_EVAL_ROLLOUTS_PER_GAME:-12}"

python -u -m examples.playbook_evolve.skill_tree_depth_v4_extend_validation \
  --root "$AB_ROOT" \
  --source_root "$SOURCE_V4_ROOT" \
  --alfworld_data "$ALFWORLD_DATA" \
  --model_path "$MODEL_PATH" \
  --eval_games_per_type "${V4_EVAL_GAMES_PER_TYPE:-5}" \
  --eval_rollouts_per_game "${V4_EVAL_ROLLOUTS_PER_GAME:-12}" \
  --data_parallel_workers "$DP_WORKERS" \
  --rollout_worker_gpus "$CUDA_VISIBLE_DEVICES" \
  --gpu_mem_util "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
  --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" \
  "$@"
