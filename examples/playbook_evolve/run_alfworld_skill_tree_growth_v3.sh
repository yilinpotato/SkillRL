#!/usr/bin/env bash
# V3 independent skill-tree growth: external evidence -> five online rounds -> held-out eval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set +u
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV:-skillRL}"; fi
set -u
source "$PROJECT_ROOT/scripts/load_private_env.sh"
source "$PROJECT_ROOT/scripts/preflight_cloud_api.sh"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export CACHE_ROOT="${CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
EXTERNAL_RAW_TRACES="${EXTERNAL_RAW_TRACES:-}"
[[ -n "$EXTERNAL_RAW_TRACES" && -f "$EXTERNAL_RAW_TRACES" ]] || {
  echo "Set EXTERNAL_RAW_TRACES to the supplied ALFWorld raw_traces.jsonl before starting V3." >&2; exit 2;
}
[[ -d "$ALFWORLD_DATA/json_2.1.1" && -f "$MODEL_PATH/config.json" ]] || {
  echo "Missing ALFWORLD_DATA or MODEL_PATH assets." >&2; exit 2;
}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
GPU_COUNT="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF{n++} END{print n+0}')"
DP_WORKERS="${DATA_PARALLEL_WORKERS:-$GPU_COUNT}"
[[ "$DP_WORKERS" == "$GPU_COUNT" ]] || { echo "DATA_PARALLEL_WORKERS must equal visible GPU count." >&2; exit 2; }
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
AB_ROOT="${AB_ROOT:-$PROJECT_ROOT/skillrl_outputs/alfworld_skill_tree_growth_v3/$(date +%Y%m%d_%H%M%S)}"
echo "[skill-tree-v3] GPUs=$CUDA_VISIBLE_DEVICES DP=$DP_WORKERS root=$AB_ROOT"
echo "[skill-tree-v3] initial=12/task, rounds=5 x 2 online rollouts/task, eval=3 games x 12 rollouts/task"
python -u -m examples.playbook_evolve.skill_tree_growth_ablation_v3 \
  --root "$AB_ROOT" --external_raw_traces "$EXTERNAL_RAW_TRACES" --alfworld_data "$ALFWORLD_DATA" \
  --model_path "$MODEL_PATH" --data_parallel_workers "$DP_WORKERS" \
  --rollout_worker_gpus "$CUDA_VISIBLE_DEVICES" --gpu_mem_util "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
  --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" "$@"
