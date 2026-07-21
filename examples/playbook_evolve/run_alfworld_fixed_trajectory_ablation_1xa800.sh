#!/usr/bin/env bash
# Direct (non-Docker) launcher for a fixed-trajectory ALFWorld ablation on one
# allocated A800.  It deliberately keeps the protocol unchanged: one TP=1
# vLLM replica executes the same fixed 36 bootstrap/evaluation rollouts per
# arm, only with lower throughput than the 4xA800 launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BASE_LAUNCHER="$SCRIPT_DIR/run_alfworld_fixed_trajectory_ablation.sh"
cd "$PROJECT_ROOT"

set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-skillRL}"
fi
set -u

source "$PROJECT_ROOT/scripts/load_private_env.sh"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# A scheduler should set this to its assigned A800.  Defaulting to 0 makes an
# interactive one-card allocation explicit rather than silently consuming all
# GPUs on a node.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VISIBLE_GPU_COUNT="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF {n++} END {print n+0}')"
if [[ "$VISIBLE_GPU_COUNT" != "1" ]]; then
  echo "This launcher requires exactly one allocated GPU; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi
export DATA_PARALLEL_WORKERS=1
export ROLLOUT_WORKER_GPUS="$CUDA_VISIBLE_DEVICES"
# The base launcher normally applies the shared local-3090 policy by looking
# at physical GPU 0.  This wrapper has already validated an explicit one-GPU
# accelerator allocation, so it must not be misclassified on a mixed node.
export COSKILL_FORCE_ACCELERATOR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

export CACHE_ROOT="${CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/skillrl_outputs}"
RUN_ID="${ABLATION_RUN_ID:-1xa800_$(date +%Y%m%d_%H%M%S)}"
export AB_ROOT="${AB_ROOT:-$OUTPUT_ROOT/alfworld_fixed_trajectory_ablation/$RUN_ID}"

if [[ ! -d "$ALFWORLD_DATA/json_2.1.1" || ! -d "$ALFWORLD_DATA/logic" ]]; then
  echo "ALFWorld data is incomplete: ALFWORLD_DATA=$ALFWORLD_DATA" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Model config is missing: MODEL_PATH=$MODEL_PATH" >&2
  exit 2
fi
shopt -s nullglob
model_weights=("$MODEL_PATH"/*.safetensors)
shopt -u nullglob
if ((${#model_weights[@]} == 0)); then
  echo "No safetensors weights found under MODEL_PATH=$MODEL_PATH" >&2
  exit 2
fi

python - <<'PY'
import torch

if torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected one CUDA device, got {torch.cuda.device_count()}")
props = torch.cuda.get_device_properties(0)
name = props.name
print(f"[1xa800-preflight] cuda:0={name} ({props.total_memory / 2**30:.1f} GiB)")
if "a800" not in name.lower():
    print(f"[1xa800-preflight] WARNING: expected A800, got {name}")
PY

# The complete protocol makes cloud calls while creating the flat/tree arms.
# Probe before reserving vLLM so a bad credential fails cheaply. Set 0 only
# for local dependency checks; a real all-phase run still needs usable cloud
# credentials when it reaches artifact construction.
if [[ "${CLOUD_PROBE:-1}" == "1" ]]; then
  python "$PROJECT_ROOT/scripts/check_cloud_bootstrap.py" \
    --environment alfworld \
    --skills-json "$PROJECT_ROOT/memory_data/alfworld/claude_style_skills.json" \
    --probe
elif [[ "${CLOUD_PROBE:-1}" != "0" ]]; then
  echo "CLOUD_PROBE must be 0 or 1, got: ${CLOUD_PROBE}" >&2
  exit 2
fi

echo "[1xa800-ablation] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[1xa800-ablation] MODEL_PATH=$MODEL_PATH"
echo "[1xa800-ablation] ALFWORLD_DATA=$ALFWORLD_DATA"
echo "[1xa800-ablation] AB_ROOT=$AB_ROOT"
echo "[1xa800-ablation] DP=1 TP=1 total_bootstrap_rollouts=36 total_eval_rollouts_per_arm=36"

if [[ "${ABLATION_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[1xa800-ablation] preflight passed; no rollout started."
  exit 0
fi

if (( $# == 0 )); then
  set -- --phase "${ABLATION_PHASE:-all}"
fi
exec bash "$BASE_LAUNCHER" "$@"
