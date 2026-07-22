#!/usr/bin/env bash
# Direct (non-Docker) launcher for a fixed-trajectory ALFWorld ablation on one
# allocated A800.  One TP=1 vLLM replica executes the formal fixed 72
# bootstrap/evaluation rollouts per arm (six games x 12 replicas).
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
# Keep the vLLM allocation request and the launcher preflight in one place.
# A previous rollout's EngineCore can take a short time to release memory, so
# wait briefly by default instead of failing inside vLLM with an opaque stack.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
GPU_MEMORY_WAIT_SECONDS="${GPU_MEMORY_WAIT_SECONDS:-300}"
GPU_MEMORY_HEADROOM_MIB="${GPU_MEMORY_HEADROOM_MIB:-512}"

# A second nohup invocation on the same allocated card cannot share an 80%
# vLLM reservation. Keep this descriptor open across the final exec so the
# advisory lock lasts for the whole controller and every child rollout.
LOCK_KEY="$(tr -c 'A-Za-z0-9' '_' <<<"$CUDA_VISIBLE_DEVICES")"
LOCK_PATH="/tmp/coskill-alfworld-fixed-ablation-${LOCK_KEY}.lock"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "Another fixed-trajectory launcher already holds $LOCK_PATH for CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES." >&2
  echo "Inspect it with: lsof $LOCK_PATH; nvidia-smi" >&2
  exit 2
fi

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

# vLLM itself rejects a replica when free memory is below gpu_memory_utilization
# of the card. Check the same condition before launching Python, report the
# occupying PIDs, and optionally wait for a just-finished EngineCore to exit.
export GPU_MEMORY_WAIT_SECONDS GPU_MEMORY_HEADROOM_MIB
python - <<'PY'
import math
import os
import subprocess
import time

gpu = os.environ["CUDA_VISIBLE_DEVICES"]
if not gpu.isdecimal():
    raise SystemExit(
        "GPU memory preflight currently requires a numeric CUDA_VISIBLE_DEVICES "
        f"entry, got {gpu!r}. Set it to the allocated physical GPU index."
    )
util = float(os.environ["VLLM_GPU_MEMORY_UTILIZATION"])
wait_seconds = int(os.environ["GPU_MEMORY_WAIT_SECONDS"])
headroom_mib = int(os.environ["GPU_MEMORY_HEADROOM_MIB"])
if not 0 < util <= 1:
    raise SystemExit(f"VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1], got {util}")
if wait_seconds < 0 or headroom_mib < 0:
    raise SystemExit("GPU_MEMORY_WAIT_SECONDS and GPU_MEMORY_HEADROOM_MIB must be >= 0")

def query(cmd):
    return subprocess.check_output(cmd, text=True).strip()

deadline = time.monotonic() + wait_seconds
while True:
    total, free = [int(x) for x in query([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]).split(",")]
    required = math.ceil(total * util) + headroom_mib
    if free >= required:
        print(
            f"[1xa800-preflight] GPU {gpu} free={free} MiB; "
            f"required={required} MiB (util={util}, headroom={headroom_mib} MiB)"
        )
        break
    apps = query([
        "nvidia-smi", f"--id={gpu}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    remaining = max(0, math.ceil(deadline - time.monotonic()))
    print(
        f"[1xa800-preflight] GPU {gpu} free={free}/{total} MiB, but vLLM needs "
        f"at least {required} MiB. Waiting up to {remaining}s for occupying processes:\n"
        f"{apps or '(nvidia-smi reports no compute process; retrying)'}",
        flush=True,
    )
    if time.monotonic() >= deadline:
        raise SystemExit(
            "GPU memory preflight timed out. Do not lower gpu_mem_util to hide a "
            "foreign workload; wait for it to finish or use the scheduler-assigned free GPU."
        )
    time.sleep(min(10, max(1, remaining)))
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
echo "[1xa800-ablation] vllm_gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION wait_seconds=$GPU_MEMORY_WAIT_SECONDS"
echo "[1xa800-ablation] DP=1 TP=1 total_bootstrap_rollouts=${ABLATION_ROLLOUTS_PER_TYPE:-12}x6=$(( ${ABLATION_ROLLOUTS_PER_TYPE:-12} * 6 )) total_eval_rollouts_per_arm=$(( ${ABLATION_ROLLOUTS_PER_TYPE:-12} * 6 ))"

if [[ "${ABLATION_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[1xa800-ablation] preflight passed; no rollout started."
  exit 0
fi

if (( $# == 0 )); then
  set -- --phase "${ABLATION_PHASE:-all}"
fi
exec bash "$BASE_LAUNCHER" --gpu_mem_util "$VLLM_GPU_MEMORY_UTILIZATION" "$@"
