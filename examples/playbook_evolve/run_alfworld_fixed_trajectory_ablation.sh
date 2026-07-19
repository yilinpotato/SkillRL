#!/usr/bin/env bash
# Fixed-trajectory ALFWorld ablations.  This script intentionally does not
# source run_alfworld_playbook_evolve_norl.sh because that launcher executes a
# training run immediately; it only mirrors its environment contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-skillRL}"
fi
set -u

source "$PROJECT_ROOT/scripts/load_private_env.sh"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1

GPU0_NAME="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if [[ "${COSKILL_FORCE_LOCAL_3090:-0}" == "1" || "$GPU0_NAME" == *"RTX 3090"* ]]; then
  # The shared local 3090 rule remains in force.
  [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == "0" ]] || {
    echo "Local ablations may only use physical GPU 0." >&2; exit 1;
  }
  export CUDA_VISIBLE_DEVICES="0"
  active="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF' || true)"
  [[ -z "$active" ]] || { echo "GPU 0 is in use: $active" >&2; exit 1; }
  RUN_ENV="local-3090"
else
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF{n++} END{print n+0}')"
    (( GPU_COUNT > 0 )) || { echo "No NVIDIA GPU detected." >&2; exit 1; }
    (( GPU_COUNT > 8 )) && GPU_COUNT=8
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
  else
    mapfile -t _visible_gpus < <(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF' | head -n 8)
    ((${#_visible_gpus[@]} > 0)) || { echo "CUDA_VISIBLE_DEVICES is empty." >&2; exit 1; }
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${_visible_gpus[*]}")"
    unset _visible_gpus
  fi
  RUN_ENV="accelerator"
fi

# Host-specific absolute paths belong in .env or exported variables, not code.
export CACHE_ROOT="${CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$CACHE_ROOT/alfworld}"
export MODEL_PATH="${MODEL_PATH:-$CACHE_ROOT/modelscope/hub/models/Qwen/Qwen3-4B-Thinking-2507}"

AB_ROOT="${AB_ROOT:-$PROJECT_ROOT/skillrl_outputs/alfworld_fixed_trajectory_ablation/$(date +%Y%m%d_%H%M%S)}"
DP_WORKERS="${DATA_PARALLEL_WORKERS:-$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | grep -c .)}"
echo "[ablation] env=$RUN_ENV GPUs=$CUDA_VISIBLE_DEVICES DP=$DP_WORKERS root=$AB_ROOT"

python -u -m examples.playbook_evolve.fixed_trajectory_ablation \
  --root "$AB_ROOT" \
  --alfworld_data "$ALFWORLD_DATA" \
  --model_path "$MODEL_PATH" \
  --data_parallel_workers "$DP_WORKERS" \
  --rollout_worker_gpus "$CUDA_VISIBLE_DEVICES" \
  "$@"
