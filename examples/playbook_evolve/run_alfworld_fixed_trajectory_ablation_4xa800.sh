#!/usr/bin/env bash
# Direct (non-Docker) launcher for the fixed-trajectory ALFWorld ablations on
# one isolated four-A800 allocation. It dispatches four independent TP=1
# vLLM rollout workers; fixed manifests and request seeds keep the 72 rollout
# trajectories per arm identical to 1/2/8-GPU executions.
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

# Read cloud configuration before preflight. The base launcher loads this too;
# doing it here lets CLOUD_PROBE fail before any GPU worker is started.
PRIVATE_ENV_FILE="${COSKILL_ENV_FILE:-$PROJECT_ROOT/.env}"
source "$PROJECT_ROOT/scripts/load_private_env.sh"
unset PRIVATE_ENV_FILE

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
VISIBLE_GPU_COUNT="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF {n++} END {print n+0}')"
if [[ "$VISIBLE_GPU_COUNT" != "4" ]]; then
  echo "This launcher requires exactly four allocated GPUs; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi
export DATA_PARALLEL_WORKERS=4
export ROLLOUT_WORKER_GPUS="$CUDA_VISIBLE_DEVICES"
export COSKILL_FORCE_ACCELERATOR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

export CACHE_ROOT="${CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/skillrl_outputs}"
RUN_ID="${ABLATION_RUN_ID:-4xa800_$(date +%Y%m%d_%H%M%S)}"
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

if torch.cuda.device_count() != 4:
    raise SystemExit(f"Expected four CUDA devices, got {torch.cuda.device_count()}")
for index in range(4):
    props = torch.cuda.get_device_properties(index)
    name = props.name
    print(f"[4xa800-preflight] cuda:{index}={name} ({props.total_memory / 2**30:.1f} GiB)")
    if "a800" not in name.lower():
        print(f"[4xa800-preflight] WARNING: expected A800, got {name}")
PY

if [[ "${CLOUD_PROBE:-1}" == "1" ]]; then
  python "$PROJECT_ROOT/scripts/check_cloud_bootstrap.py" \
    --environment alfworld \
    --skills-json "$PROJECT_ROOT/memory_data/alfworld/claude_style_skills.json" \
    --probe
elif [[ "${CLOUD_PROBE:-1}" != "0" ]]; then
  echo "CLOUD_PROBE must be 0 or 1, got: ${CLOUD_PROBE}" >&2
  exit 2
fi

echo "[4xa800-ablation] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[4xa800-ablation] MODEL_PATH=$MODEL_PATH"
echo "[4xa800-ablation] ALFWORLD_DATA=$ALFWORLD_DATA"
echo "[4xa800-ablation] AB_ROOT=$AB_ROOT"
echo "[4xa800-ablation] DP=4 TP=1 total_eval_rollouts_per_arm=${ABLATION_ROLLOUTS_PER_TYPE:-12}x6=$(( ${ABLATION_ROLLOUTS_PER_TYPE:-12} * 6 ))"

if [[ "${ABLATION_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[4xa800-ablation] preflight passed; no rollout started."
  exit 0
fi

# With no explicit arguments, run the complete resumable pipeline. To resume a
# phase manually, for example: `..._4xa800.sh --phase evaluate` with the same
# AB_ROOT. The controller rejects old frozen traces without its empty-bank
# protocol instead of silently reusing a contaminated corpus.
if (( $# == 0 )); then
  set -- --phase "${ABLATION_PHASE:-all}"
fi
exec bash "$BASE_LAUNCHER" "$@"
