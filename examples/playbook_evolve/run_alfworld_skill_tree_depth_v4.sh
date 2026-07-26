#!/usr/bin/env bash
# V4: frozen 12-trace evidence -> monotonic L1-L5 expansion -> shared held-out evaluation.
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
source "$PROJECT_ROOT/scripts/preflight_cloud_api.sh"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export CACHE_ROOT="${CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

EXTERNAL_RAW_TRACES="${EXTERNAL_RAW_TRACES:-}"
[[ -n "$EXTERNAL_RAW_TRACES" && -f "$EXTERNAL_RAW_TRACES" ]] || {
  echo "Set EXTERNAL_RAW_TRACES to the ALFWorld raw_traces.jsonl corpus." >&2
  exit 2
}
[[ -d "$ALFWORLD_DATA/json_2.1.1" && -f "$MODEL_PATH/config.json" ]] || {
  echo "Missing ALFWORLD_DATA or MODEL_PATH assets." >&2
  exit 2
}

GPU_COUNT="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF{n++} END{print n+0}')"
DP_WORKERS="${DATA_PARALLEL_WORKERS:-$GPU_COUNT}"
[[ "$GPU_COUNT" -ge 1 && "$GPU_COUNT" -le 2 ]] || {
  echo "V4 supports one or two visible GPUs; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
}
[[ "$DP_WORKERS" == "$GPU_COUNT" ]] || {
  echo "DATA_PARALLEL_WORKERS must equal the visible GPU count." >&2
  exit 2
}
TORCH_GPU_COUNT="$(
  python -c 'import torch; print(torch.cuda.device_count())'
)"
[[ "$TORCH_GPU_COUNT" == "$GPU_COUNT" ]] || {
  echo "CUDA mask resolves to $TORCH_GPU_COUNT runtime device(s), expected $GPU_COUNT from CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES." >&2
  echo "Use the logical indices shown inside this container (a one-GPU container normally uses CUDA_VISIBLE_DEVICES=0)." >&2
  exit 2
}

VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
AB_ROOT="${AB_ROOT:-$PROJECT_ROOT/skillrl_outputs/alfworld_skill_tree_depth_v4/$(date +%Y%m%d_%H%M%S)}"
echo "[skill-tree-v4] GPUs=$CUDA_VISIBLE_DEVICES DP=$DP_WORKERS root=$AB_ROOT"
echo "[skill-tree-v4] 12 full traces/task (6 success+6 failure); no online growth"
echo "[skill-tree-v4] L1->L5 monotonic extension; no node/char ceiling; eval=3 games/task x 12 rollouts"
echo "[skill-tree-v4] local_max_model_len=${V4_LOCAL_MAX_MODEL_LEN:-16384}; any prompt trim invalidates the arm"

python -u -m examples.playbook_evolve.skill_tree_depth_ablation_v4 \
  --root "$AB_ROOT" \
  --external_raw_traces "$EXTERNAL_RAW_TRACES" \
  --alfworld_data "$ALFWORLD_DATA" \
  --model_path "$MODEL_PATH" \
  --data_parallel_workers "$DP_WORKERS" \
  --rollout_worker_gpus "$CUDA_VISIBLE_DEVICES" \
  --gpu_mem_util "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
  --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" \
  "$@"
