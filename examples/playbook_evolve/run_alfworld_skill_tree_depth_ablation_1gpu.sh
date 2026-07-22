#!/usr/bin/env bash
# Generic single-GPU entrypoint for the independent ALFWorld L0-L5 experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF {n++} END {print n+0}')"
if [[ "$GPU_COUNT" != "1" ]]; then
  echo "Expected exactly one visible GPU; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
export DATA_PARALLEL_WORKERS=1
export ROLLOUT_WORKER_GPUS="$CUDA_VISIBLE_DEVICES"
export TENSOR_PARALLEL_SIZE=1

exec bash "$SCRIPT_DIR/run_alfworld_skill_tree_depth_ablation.sh" "$@"
