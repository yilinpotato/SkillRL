#!/usr/bin/env bash
# Explicit four-A800 entrypoint for the compression-off no-RL arm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
GPU_COUNT="$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF {n++} END {print n+0}')"
if [[ "$GPU_COUNT" != "4" ]]; then
  echo "Expected exactly four visible GPUs; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
export DATA_PARALLEL_WORKERS=4
export ROLLOUT_WORKER_GPUS="$CUDA_VISIBLE_DEVICES"
export TENSOR_PARALLEL_SIZE=1
# CUDA Graph changes execution scheduling only; prompts, sampling seeds, reward,
# cloud updates, and the 100x72 experiment protocol remain unchanged.
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

exec bash "$SCRIPT_DIR/run_alfworld_trace_compression_off_norl.sh" "$@"
